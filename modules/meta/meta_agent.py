# ─────────────────────────────────────────────────────────────
# File: modules/meta/meta_agent.py
# Enhanced with InfoBus integration & intelligent automation
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from enum import Enum

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class MetaMode(Enum):
    """Meta agent operational modes"""
    TRAINING = "training"
    LIVE_TRADING = "live_trading" 
    RETRAINING = "retraining"
    EVALUATION = "evaluation"
    EMERGENCY_STOP = "emergency_stop"


class MetaAgent(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced meta agent with InfoBus integration and intelligent automation.
    Manages the entire training/trading lifecycle with automatic decision making
    for when to retrain, when to go live, and when to stop trading.
    """
    
    def __init__(self, window: int = 20, debug: bool = True, 
                 profit_target: float = 150.0, 
                 retrain_threshold: float = -50.0,
                 emergency_threshold: float = -100.0,
                 confidence_threshold: float = 0.7,
                 **kwargs):
        
        # Enhanced configuration
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            health_check_interval=60,
            performance_window=100,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.window = window
        self.profit_target = profit_target
        self.retrain_threshold = retrain_threshold
        self.emergency_threshold = emergency_threshold
        self.confidence_threshold = confidence_threshold
        
        # Operational state
        self.current_mode = MetaMode.EVALUATION
        self.mode_start_time = datetime.datetime.now()
        self.consecutive_losses = 0
        self.mode_transitions = deque(maxlen=50)
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.peak_pnl = 0.0
        self.drawdown_pct = 0.0
        self.win_streak = 0
        self.loss_streak = 0
        
        # Training performance
        self.training_episodes = 0
        self.training_start_time = None
        self.best_training_reward = -np.inf
        self.training_convergence_count = 0
        
        # Automation thresholds and conditions
        self.automation_config = {
            'min_training_episodes': 100,
            'convergence_episodes': 10,
            'improvement_threshold': 0.05,
            'live_evaluation_period': 3600,  # 1 hour
            'retrain_cooldown': 7200,  # 2 hours
            'emergency_cooldown': 1800,  # 30 minutes
            'confidence_decay': 0.98,
            'performance_smoothing': 0.95
        }
        
        # State tracking
        self.system_confidence = 0.5
        self.last_retrain_time = None
        self.last_emergency_time = None
        self.evaluation_start_time = None
        
        # Decision history for learning
        self.decision_history = deque(maxlen=100)
        self.automation_metrics = {
            'total_mode_switches': 0,
            'successful_live_sessions': 0,
            'emergency_stops': 0,
            'retraining_sessions': 0,
            'avg_training_duration': 0.0,
            'avg_live_duration': 0.0,
            'automation_accuracy': 0.0
        }
        
        # Enhanced logging with rotation
        self.logger = RotatingLogger(
            "MetaAgent",
            "logs/strategy/meta/meta_agent.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("MetaAgent")
        
        self.log_operator_info(
            "🤖 Enhanced Meta Agent initialized",
            profit_target=f"€{profit_target}",
            retrain_threshold=f"€{retrain_threshold}",
            emergency_threshold=f"€{emergency_threshold}",
            mode=self.current_mode.value,
            automation_enabled=True
        )
    
    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Reset performance tracking
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.peak_pnl = 0.0
        self.drawdown_pct = 0.0
        self.consecutive_losses = 0
        self.win_streak = 0
        self.loss_streak = 0
        
        # Reset mode state
        self.current_mode = MetaMode.EVALUATION
        self.mode_start_time = datetime.datetime.now()
        self.mode_transitions.clear()
        
        # Reset training state
        self.training_episodes = 0
        self.training_start_time = None
        self.best_training_reward = -np.inf
        self.training_convergence_count = 0
        
        # Reset automation state
        self.system_confidence = 0.5
        self.decision_history.clear()
        
        self.log_operator_info("🔄 Meta Agent reset - all state cleared")
    
    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and intelligent automation"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract context and performance data
        context = extract_standard_context(info_bus)
        self._update_performance_from_info_bus(info_bus)
        
        # Core automation logic
        self._evaluate_system_health(info_bus, context)
        self._update_system_confidence(info_bus, context)
        decision = self._make_automation_decision(info_bus, context)
        
        if decision:
            self._execute_automation_decision(decision, info_bus, context)
        
        # Update analytics and tracking
        self._update_automation_metrics()
        self._track_decision_quality()
        
        # Publish automation status to InfoBus
        self._publish_automation_status(info_bus)
    
    def _process_legacy_step(self, **kwargs):
        """Fallback processing for backward compatibility"""
        pnl = kwargs.get('pnl', 0.0)
        
        if np.isnan(pnl):
            self.log_operator_error("NaN PnL received, setting to 0")
            pnl = 0.0
        
        self._update_pnl_metrics(pnl)
        
        # Basic automation without InfoBus
        if self.current_mode == MetaMode.LIVE_TRADING:
            if pnl < self.emergency_threshold:
                self._transition_to_mode(MetaMode.EMERGENCY_STOP, "Emergency PnL threshold")
            elif self.daily_pnl < self.retrain_threshold:
                self._transition_to_mode(MetaMode.RETRAINING, "Retrain PnL threshold")
    
    def _update_performance_from_info_bus(self, info_bus: InfoBus):
        """Extract and update performance metrics from InfoBus"""
        
        # Extract PnL information
        recent_trades = info_bus.get('recent_trades', [])
        for trade in recent_trades:
            pnl = trade.get('pnl', 0.0)
            self._update_pnl_metrics(pnl)
        
        # Extract portfolio metrics
        portfolio_metrics = info_bus.get('portfolio_metrics', {})
        self.daily_pnl = portfolio_metrics.get('daily_pnl', self.daily_pnl)
        self.drawdown_pct = portfolio_metrics.get('max_drawdown_pct', 0.0)
        
        # Extract training metrics if in training mode
        if self.current_mode == MetaMode.TRAINING:
            training_metrics = info_bus.get('training_metrics', {})
            if training_metrics:
                self._update_training_metrics(training_metrics)
    
    def _update_pnl_metrics(self, pnl: float):
        """Update PnL-based metrics and streaks"""
        
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            if self.consecutive_losses > 0:
                self.log_operator_info(
                    f"💰 Loss streak broken after {self.consecutive_losses} losses",
                    profit=f"€{pnl:.2f}",
                    win_streak=self.win_streak
                )
            self.consecutive_losses = 0
        elif pnl < 0:
            self.loss_streak += 1
            self.win_streak = 0
            self.consecutive_losses += 1
            self.log_operator_warning(
                f"📉 Loss recorded",
                loss=f"€{pnl:.2f}",
                consecutive_losses=self.consecutive_losses,
                loss_streak=self.loss_streak
            )
        
        self.session_pnl += pnl
        self.daily_pnl += pnl
        
        # Update peak and drawdown
        if self.daily_pnl > self.peak_pnl:
            self.peak_pnl = self.daily_pnl
        
        if self.peak_pnl > 0:
            self.drawdown_pct = max(0, (self.peak_pnl - self.daily_pnl) / self.peak_pnl * 100)
    
    def _update_training_metrics(self, training_metrics: Dict[str, Any]):
        """Update training-specific metrics"""
        
        episode_reward = training_metrics.get('episode_reward_mean', 0)
        self.training_episodes = training_metrics.get('episodes', self.training_episodes)
        
        # Track training improvement
        if episode_reward > self.best_training_reward:
            improvement = episode_reward - self.best_training_reward
            if improvement > self.automation_config['improvement_threshold']:
                self.training_convergence_count += 1
                self.log_operator_info(
                    f"📈 Training improvement detected",
                    episode_reward=f"{episode_reward:.3f}",
                    improvement=f"{improvement:.3f}",
                    convergence_count=self.training_convergence_count
                )
            self.best_training_reward = episode_reward
        
        # Check for training convergence
        if self.training_convergence_count >= self.automation_config['convergence_episodes']:
            if self.training_episodes >= self.automation_config['min_training_episodes']:
                self.log_operator_info(
                    "🎯 Training convergence achieved",
                    episodes=self.training_episodes,
                    best_reward=f"{self.best_training_reward:.3f}",
                    convergence_episodes=self.training_convergence_count
                )
    
    def _evaluate_system_health(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Comprehensive system health evaluation"""
        
        health_score = 1.0
        health_factors = {}
        
        # Performance health
        if self.daily_pnl < self.emergency_threshold:
            health_factors['emergency_pnl'] = 0.0
            health_score *= 0.1
        elif self.daily_pnl < self.retrain_threshold:
            health_factors['poor_pnl'] = 0.3
            health_score *= 0.3
        elif self.daily_pnl > self.profit_target * 0.5:
            health_factors['good_pnl'] = 1.0
        else:
            health_factors['neutral_pnl'] = 0.7
            health_score *= 0.7
        
        # Drawdown health
        if self.drawdown_pct > 20:
            health_factors['high_drawdown'] = 0.2
            health_score *= 0.2
        elif self.drawdown_pct > 10:
            health_factors['medium_drawdown'] = 0.5
            health_score *= 0.5
        else:
            health_factors['low_drawdown'] = 1.0
        
        # Loss streak health
        if self.consecutive_losses > 10:
            health_factors['long_loss_streak'] = 0.1
            health_score *= 0.1
        elif self.consecutive_losses > 5:
            health_factors['medium_loss_streak'] = 0.4
            health_score *= 0.4
        else:
            health_factors['manageable_losses'] = 1.0
        
        # Market regime health
        regime = context.get('regime', 'unknown')
        volatility = context.get('volatility_level', 'medium')
        
        if regime == 'volatile' and volatility == 'extreme':
            health_factors['extreme_market'] = 0.3
            health_score *= 0.3
        elif regime == 'ranging' and volatility == 'low':
            health_factors['difficult_market'] = 0.6
            health_score *= 0.6
        else:
            health_factors['normal_market'] = 1.0
        
        # Store health assessment
        self._update_performance_metric('system_health', health_score)
        
        if health_score < 0.3:
            self.log_operator_error(
                "🚨 Critical system health detected",
                health_score=f"{health_score:.3f}",
                factors=health_factors,
                daily_pnl=f"€{self.daily_pnl:.2f}",
                drawdown=f"{self.drawdown_pct:.1f}%"
            )
        elif health_score < 0.6:
            self.log_operator_warning(
                "⚠️ Poor system health",
                health_score=f"{health_score:.3f}",
                factors=health_factors
            )
    
    def _update_system_confidence(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Update system confidence based on recent performance"""
        
        # Performance-based confidence
        performance_confidence = 0.5
        if self.daily_pnl > self.profit_target * 0.8:
            performance_confidence = 0.9
        elif self.daily_pnl > self.profit_target * 0.5:
            performance_confidence = 0.8
        elif self.daily_pnl > 0:
            performance_confidence = 0.7
        elif self.daily_pnl > self.retrain_threshold:
            performance_confidence = 0.4
        else:
            performance_confidence = 0.2
        
        # Training-based confidence
        training_confidence = 0.5
        if self.current_mode == MetaMode.TRAINING:
            if self.training_convergence_count >= self.automation_config['convergence_episodes']:
                training_confidence = 0.9
            elif self.training_episodes >= self.automation_config['min_training_episodes']:
                training_confidence = 0.7
        
        # Market-based confidence
        market_confidence = 0.7
        regime = context.get('regime', 'unknown')
        vol_level = context.get('volatility_level', 'medium')
        
        if regime == 'trending' and vol_level in ['low', 'medium']:
            market_confidence = 0.9
        elif regime == 'volatile' and vol_level == 'extreme':
            market_confidence = 0.3
        
        # Combine confidences
        new_confidence = (0.5 * performance_confidence + 
                         0.3 * training_confidence + 
                         0.2 * market_confidence)
        
        # Apply exponential smoothing
        self.system_confidence = (self.automation_config['confidence_decay'] * self.system_confidence + 
                                 (1 - self.automation_config['confidence_decay']) * new_confidence)
        
        self._update_performance_metric('system_confidence', self.system_confidence)
    
    def _make_automation_decision(self, info_bus: InfoBus, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Intelligent automation decision making"""
        
        current_time = datetime.datetime.now()
        mode_duration = (current_time - self.mode_start_time).total_seconds()
        
        # Emergency stop conditions
        if (self.daily_pnl <= self.emergency_threshold or 
            self.consecutive_losses >= 15 or 
            self.drawdown_pct >= 25):
            
            if self._check_cooldown(self.last_emergency_time, self.automation_config['emergency_cooldown']):
                return {
                    'action': 'emergency_stop',
                    'reason': f'Emergency conditions: PnL=€{self.daily_pnl:.2f}, Losses={self.consecutive_losses}, DD={self.drawdown_pct:.1f}%',
                    'priority': 'critical'
                }
        
        # Mode-specific decision logic
        if self.current_mode == MetaMode.EVALUATION:
            return self._evaluate_mode_decision(info_bus, context, mode_duration)
        elif self.current_mode == MetaMode.TRAINING:
            return self._training_mode_decision(info_bus, context, mode_duration)
        elif self.current_mode == MetaMode.LIVE_TRADING:
            return self._live_trading_decision(info_bus, context, mode_duration)
        elif self.current_mode == MetaMode.RETRAINING:
            return self._retraining_decision(info_bus, context, mode_duration)
        
        return None
    
    def _evaluate_mode_decision(self, info_bus: InfoBus, context: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Decision logic for evaluation mode"""
        
        # Minimum evaluation period
        if mode_duration < 300:  # 5 minutes
            return None
        
        # If system confidence is high, consider live trading
        if self.system_confidence >= self.confidence_threshold:
            if self.daily_pnl > 0:  # Profitable day so far
                return {
                    'action': 'start_live_trading',
                    'reason': f'High confidence ({self.system_confidence:.3f}) and profitable',
                    'priority': 'normal'
                }
        
        # If performance is poor, start training
        if self.daily_pnl < self.retrain_threshold * 0.5:
            return {
                'action': 'start_training',
                'reason': f'Poor performance requires training: €{self.daily_pnl:.2f}',
                'priority': 'high'
            }
        
        # Extended evaluation - start with conservative training
        if mode_duration > self.automation_config['live_evaluation_period']:
            return {
                'action': 'start_training',
                'reason': 'Evaluation period complete, starting training',
                'priority': 'normal'
            }
        
        return None
    
    def _training_mode_decision(self, info_bus: InfoBus, context: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Decision logic for training mode"""
        
        # Check if training has converged
        if (self.training_convergence_count >= self.automation_config['convergence_episodes'] and
            self.training_episodes >= self.automation_config['min_training_episodes']):
            
            return {
                'action': 'start_evaluation',
                'reason': f'Training converged: {self.training_episodes} episodes, {self.training_convergence_count} improvements',
                'priority': 'normal'
            }
        
        # Max training time protection
        if mode_duration > 14400:  # 4 hours
            return {
                'action': 'start_evaluation',
                'reason': 'Max training time reached',
                'priority': 'high'
            }
        
        return None
    
    def _live_trading_decision(self, info_bus: InfoBus, context: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Decision logic for live trading mode"""
        
        # Retrain if performance degrades
        if self.daily_pnl <= self.retrain_threshold:
            if self._check_cooldown(self.last_retrain_time, self.automation_config['retrain_cooldown']):
                return {
                    'action': 'start_retraining',
                    'reason': f'Performance below threshold: €{self.daily_pnl:.2f}',
                    'priority': 'high'
                }
        
        # Stop if confidence drops significantly
        if self.system_confidence < 0.4:
            return {
                'action': 'start_evaluation',
                'reason': f'Low system confidence: {self.system_confidence:.3f}',
                'priority': 'normal'
            }
        
        # Profit taking - if target achieved, evaluate new conditions
        if self.daily_pnl >= self.profit_target:
            return {
                'action': 'start_evaluation',
                'reason': f'Profit target achieved: €{self.daily_pnl:.2f}',
                'priority': 'normal'
            }
        
        return None
    
    def _retraining_decision(self, info_bus: InfoBus, context: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Decision logic for retraining mode"""
        
        # Similar to training but more aggressive thresholds
        if (self.training_convergence_count >= max(3, self.automation_config['convergence_episodes'] // 2) and
            self.training_episodes >= self.automation_config['min_training_episodes'] // 2):
            
            return {
                'action': 'start_evaluation',
                'reason': f'Retraining complete: {self.training_episodes} episodes',
                'priority': 'normal'
            }
        
        # Faster timeout for retraining
        if mode_duration > 7200:  # 2 hours
            return {
                'action': 'start_evaluation',
                'reason': 'Max retraining time reached',
                'priority': 'high'
            }
        
        return None
    
    def _check_cooldown(self, last_time: Optional[datetime.datetime], cooldown_seconds: float) -> bool:
        """Check if enough time has passed since last action"""
        if last_time is None:
            return True
        
        time_since = (datetime.datetime.now() - last_time).total_seconds()
        return time_since >= cooldown_seconds
    
    def _execute_automation_decision(self, decision: Dict[str, Any], info_bus: InfoBus, context: Dict[str, Any]):
        """Execute automation decision"""
        
        action = decision['action']
        reason = decision['reason']
        priority = decision['priority']
        
        self.log_operator_info(
            f"🎯 Automation decision: {action}",
            reason=reason,
            priority=priority,
            current_mode=self.current_mode.value,
            confidence=f"{self.system_confidence:.3f}",
            daily_pnl=f"€{self.daily_pnl:.2f}"
        )
        
        # Execute the decision
        if action == 'emergency_stop':
            self._transition_to_mode(MetaMode.EMERGENCY_STOP, reason)
            self.last_emergency_time = datetime.datetime.now()
            
        elif action == 'start_training':
            self._transition_to_mode(MetaMode.TRAINING, reason)
            self.training_episodes = 0
            self.training_start_time = datetime.datetime.now()
            self.best_training_reward = -np.inf
            self.training_convergence_count = 0
            
        elif action == 'start_retraining':
            self._transition_to_mode(MetaMode.RETRAINING, reason)
            self.last_retrain_time = datetime.datetime.now()
            self.training_episodes = 0
            self.training_start_time = datetime.datetime.now()
            self.best_training_reward = -np.inf
            self.training_convergence_count = 0
            
        elif action == 'start_live_trading':
            self._transition_to_mode(MetaMode.LIVE_TRADING, reason)
            
        elif action == 'start_evaluation':
            self._transition_to_mode(MetaMode.EVALUATION, reason)
            self.evaluation_start_time = datetime.datetime.now()
        
        # Record decision for learning
        self.decision_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action,
            'reason': reason,
            'context': context.copy(),
            'daily_pnl': self.daily_pnl,
            'confidence': self.system_confidence,
            'mode_before': self.current_mode.value
        })
        
        # Update automation metrics
        self.automation_metrics['total_mode_switches'] += 1
        
    def _transition_to_mode(self, new_mode: MetaMode, reason: str):
        """Transition to a new operational mode"""
        
        old_mode = self.current_mode
        old_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        # Record transition
        transition = {
            'timestamp': datetime.datetime.now().isoformat(),
            'from_mode': old_mode.value,
            'to_mode': new_mode.value,
            'reason': reason,
            'duration': old_duration,
            'daily_pnl': self.daily_pnl,
            'confidence': self.system_confidence
        }
        self.mode_transitions.append(transition)
        
        # Update state
        self.current_mode = new_mode
        self.mode_start_time = datetime.datetime.now()
        
        # Update automation metrics based on mode
        if new_mode == MetaMode.LIVE_TRADING:
            if old_mode == MetaMode.TRAINING:
                self.automation_metrics['avg_training_duration'] = (
                    (self.automation_metrics['avg_training_duration'] * 
                     self.automation_metrics['retraining_sessions'] + old_duration) /
                    (self.automation_metrics['retraining_sessions'] + 1)
                )
                self.automation_metrics['retraining_sessions'] += 1
        
        elif new_mode == MetaMode.EMERGENCY_STOP:
            self.automation_metrics['emergency_stops'] += 1
            if old_mode == MetaMode.LIVE_TRADING:
                self.automation_metrics['avg_live_duration'] = (
                    (self.automation_metrics['avg_live_duration'] * 
                     self.automation_metrics['successful_live_sessions'] + old_duration) /
                    (self.automation_metrics['successful_live_sessions'] + 1)
                )
        
        elif old_mode == MetaMode.LIVE_TRADING and self.daily_pnl > 0:
            self.automation_metrics['successful_live_sessions'] += 1
        
        self.log_operator_info(
            f"🔄 Mode transition: {old_mode.value} → {new_mode.value}",
            reason=reason,
            duration=f"{old_duration:.0f}s",
            daily_pnl=f"€{self.daily_pnl:.2f}",
            confidence=f"{self.system_confidence:.3f}"
        )
    
    def _update_automation_metrics(self):
        """Update automation performance metrics"""
        
        # Calculate automation accuracy based on decision outcomes
        if len(self.decision_history) >= 5:
            recent_decisions = list(self.decision_history)[-5:]
            successful_decisions = 0
            
            for decision in recent_decisions:
                # Simple success heuristic: if PnL improved after decision
                if self.daily_pnl > decision['daily_pnl']:
                    successful_decisions += 1
            
            self.automation_metrics['automation_accuracy'] = successful_decisions / len(recent_decisions)
            
            self._update_performance_metric('automation_accuracy', 
                                          self.automation_metrics['automation_accuracy'])
    
    def _track_decision_quality(self):
        """Track quality of automation decisions"""
        
        if len(self.decision_history) < 2:
            return
        
        # Analyze recent decision outcomes
        recent_decision = self.decision_history[-1]
        
        # Quality factors
        quality_score = 0.5
        
        # Time since decision
        decision_time = datetime.datetime.fromisoformat(recent_decision['timestamp'])
        time_since = (datetime.datetime.now() - decision_time).total_seconds()
        
        # If PnL improved significantly after decision
        pnl_improvement = self.daily_pnl - recent_decision['daily_pnl']
        if pnl_improvement > 10:  # €10 improvement
            quality_score += 0.3
        elif pnl_improvement < -10:  # €10 deterioration
            quality_score -= 0.3
        
        # If confidence improved
        confidence_improvement = self.system_confidence - recent_decision['confidence']
        if confidence_improvement > 0.1:
            quality_score += 0.2
        elif confidence_improvement < -0.1:
            quality_score -= 0.2
        
        quality_score = max(0, min(1, quality_score))
        
        self._update_performance_metric('decision_quality', quality_score)
    
    def _publish_automation_status(self, info_bus: InfoBus):
        """Publish automation status to InfoBus"""
        
        automation_status = {
            'meta_mode': self.current_mode.value,
            'system_confidence': self.system_confidence,
            'daily_pnl': self.daily_pnl,
            'session_pnl': self.session_pnl,
            'drawdown_pct': self.drawdown_pct,
            'consecutive_losses': self.consecutive_losses,
            'win_streak': self.win_streak,
            'loss_streak': self.loss_streak,
            'automation_metrics': self.automation_metrics.copy(),
            'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds(),
            'last_transition': self.mode_transitions[-1] if self.mode_transitions else None
        }
        
        InfoBusUpdater.update_meta_status(info_bus, automation_status)
    
    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE FOR SYSTEM CONTROL
    # ═══════════════════════════════════════════════════════════════════
    
    def force_mode_transition(self, new_mode: str, reason: str = "Manual override") -> bool:
        """Force a mode transition (for external control)"""
        try:
            mode_enum = MetaMode(new_mode)
            self._transition_to_mode(mode_enum, f"Manual: {reason}")
            return True
        except ValueError:
            self.log_operator_error(f"Invalid mode: {new_mode}")
            return False
    
    def get_automation_recommendations(self) -> Dict[str, Any]:
        """Get current automation recommendations"""
        
        recommendations = []
        
        if self.current_mode == MetaMode.LIVE_TRADING:
            if self.daily_pnl <= self.retrain_threshold:
                recommendations.append({
                    'action': 'consider_retraining',
                    'reason': f'PnL below threshold: €{self.daily_pnl:.2f}',
                    'urgency': 'high'
                })
            
            if self.system_confidence < 0.5:
                recommendations.append({
                    'action': 'reduce_position_size',
                    'reason': f'Low confidence: {self.system_confidence:.3f}',
                    'urgency': 'medium'
                })
        
        elif self.current_mode == MetaMode.TRAINING:
            if self.training_convergence_count >= self.automation_config['convergence_episodes']:
                recommendations.append({
                    'action': 'consider_live_trading',
                    'reason': f'Training converged after {self.training_episodes} episodes',
                    'urgency': 'medium'
                })
        
        return {
            'recommendations': recommendations,
            'current_mode': self.current_mode.value,
            'system_confidence': self.system_confidence,
            'daily_performance': f"€{self.daily_pnl:.2f}",
            'automation_active': True
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'operational_status': {
                'mode': self.current_mode.value,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds(),
                'system_confidence': self.system_confidence,
                'automation_enabled': True
            },
            'performance_status': {
                'daily_pnl': self.daily_pnl,
                'session_pnl': self.session_pnl,
                'peak_pnl': self.peak_pnl,
                'drawdown_pct': self.drawdown_pct,
                'profit_target': self.profit_target,
                'progress_pct': (self.daily_pnl / self.profit_target) * 100
            },
            'trading_metrics': {
                'consecutive_losses': self.consecutive_losses,
                'win_streak': self.win_streak,
                'loss_streak': self.loss_streak,
                'trades_today': self._trades_processed
            },
            'automation_metrics': self.automation_metrics.copy(),
            'thresholds': {
                'retrain_threshold': self.retrain_threshold,
                'emergency_threshold': self.emergency_threshold,
                'confidence_threshold': self.confidence_threshold
            },
            'recent_transitions': list(self.mode_transitions)[-5:]
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # OBSERVATION AND ACTION METHODS (Enhanced)
    # ═══════════════════════════════════════════════════════════════════
    
    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with automation state"""
        
        try:
            # Base performance components
            avg_pnl = self.daily_pnl / max(self._trades_processed, 1)
            win_rate = self._get_win_rate()
            
            # Recent performance
            recent_pnl = self.session_pnl
            
            # Automation state
            mode_encoded = {
                MetaMode.EVALUATION: 0.0,
                MetaMode.TRAINING: 0.2,
                MetaMode.RETRAINING: 0.4,
                MetaMode.LIVE_TRADING: 0.6,
                MetaMode.EMERGENCY_STOP: 0.8
            }.get(self.current_mode, 0.0)
            
            # System health indicators
            drawdown_norm = self.drawdown_pct / 100.0
            confidence_norm = self.system_confidence
            loss_streak_norm = min(self.consecutive_losses / 10.0, 1.0)
            
            observation = np.array([
                avg_pnl / 100.0,       # Normalized average PnL
                win_rate,              # Win rate [0,1]
                recent_pnl / 100.0,    # Normalized recent PnL
                mode_encoded,          # Current automation mode
                confidence_norm,       # System confidence
                drawdown_norm,         # Normalized drawdown
                loss_streak_norm,      # Normalized loss streak
                float(len(self.decision_history)) / 100.0  # Decision history depth
            ], dtype=np.float32)
            
            # Validate observation
            if np.any(np.isnan(observation)):
                self.log_operator_error(f"NaN in observation: {observation}")
                observation = np.nan_to_num(observation)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)
    
    def get_intensity(self, instrument: str) -> float:
        """Enhanced intensity calculation with automation awareness"""
        
        try:
            # Base intensity from mode
            base_intensity = {
                MetaMode.EMERGENCY_STOP: 0.0,
                MetaMode.EVALUATION: 0.3,
                MetaMode.TRAINING: 0.1,  # Low intensity during training
                MetaMode.RETRAINING: 0.2,
                MetaMode.LIVE_TRADING: 0.8
            }.get(self.current_mode, 0.5)
            
            # Adjust by confidence
            confidence_multiplier = 0.5 + (self.system_confidence * 0.5)
            
            # Adjust by performance
            if self.daily_pnl > self.profit_target * 0.8:
                performance_multiplier = 1.2
            elif self.daily_pnl > 0:
                performance_multiplier = 1.0
            elif self.daily_pnl > self.retrain_threshold:
                performance_multiplier = 0.7
            else:
                performance_multiplier = 0.3
            
            # Adjust by consecutive losses
            loss_penalty = max(0.3, 1.0 - (self.consecutive_losses * 0.05))
            
            # Calculate final intensity
            intensity = base_intensity * confidence_multiplier * performance_multiplier * loss_penalty
            
            # Clamp to safe range
            intensity = np.clip(intensity, 0.0, 1.0)
            
            self.log_operator_debug(
                f"Intensity calculated for {instrument}",
                intensity=f"{intensity:.3f}",
                mode=self.current_mode.value,
                confidence=f"{self.system_confidence:.3f}",
                daily_pnl=f"€{self.daily_pnl:.2f}"
            )
            
            return float(intensity)
            
        except Exception as e:
            self.log_operator_error(f"Intensity calculation failed for {instrument}: {e}")
            return 0.0
    
    # ═══════════════════════════════════════════════════════════════════
    # LEGACY COMPATIBILITY METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def step(self, pnl: float = 0.0):
        """Legacy step method for backward compatibility"""
        self._process_legacy_step(pnl=pnl)
    
    def record(self, pnl: float):
        """Legacy record method for backward compatibility"""
        self._update_pnl_metrics(pnl)
    
    def get_meta_report(self) -> str:
        """Generate operator-friendly meta agent report"""
        
        # Mode status with emoji
        mode_emoji = {
            MetaMode.EVALUATION: "🔍",
            MetaMode.TRAINING: "🎓",
            MetaMode.RETRAINING: "🔄",
            MetaMode.LIVE_TRADING: "💰",
            MetaMode.EMERGENCY_STOP: "🚨"
        }
        
        # Performance status
        if self.daily_pnl >= self.profit_target:
            perf_status = "🚀 Target Achieved"
        elif self.daily_pnl > self.profit_target * 0.5:
            perf_status = "✅ On Track"
        elif self.daily_pnl > 0:
            perf_status = "📈 Profitable"
        elif self.daily_pnl > self.retrain_threshold:
            perf_status = "⚠️ Below Target"
        else:
            perf_status = "📉 Poor Performance"
        
        mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        return f"""
🤖 META AGENT STATUS
═══════════════════════════════════════
{mode_emoji.get(self.current_mode, '❓')} Mode: {self.current_mode.value.upper()}
⏱️ Duration: {mode_duration/60:.1f} minutes
🎯 Confidence: {self.system_confidence:.3f}

💰 DAILY PERFORMANCE
• PnL: €{self.daily_pnl:.2f} / €{self.profit_target:.2f} ({(self.daily_pnl/self.profit_target)*100:.1f}%)
• Status: {perf_status}
• Peak: €{self.peak_pnl:.2f}
• Drawdown: {self.drawdown_pct:.1f}%

📊 TRADING METRICS
• Win Streak: {self.win_streak}
• Loss Streak: {self.loss_streak}
• Consecutive Losses: {self.consecutive_losses}
• Total Trades: {self._trades_processed}

🔄 AUTOMATION METRICS
• Mode Switches: {self.automation_metrics['total_mode_switches']}
• Successful Sessions: {self.automation_metrics['successful_live_sessions']}
• Emergency Stops: {self.automation_metrics['emergency_stops']}
• Automation Accuracy: {self.automation_metrics['automation_accuracy']:.1%}

🎓 TRAINING STATUS
• Episodes: {self.training_episodes}
• Convergence Count: {self.training_convergence_count}
• Best Reward: {self.best_training_reward:.3f}

⚙️ THRESHOLDS
• Retrain: €{self.retrain_threshold:.2f}
• Emergency: €{self.emergency_threshold:.2f}
• Confidence: {self.confidence_threshold:.3f}

🔮 RECENT TRANSITIONS
{chr(10).join([f"• {t['from_mode']} → {t['to_mode']}: {t['reason']}" for t in list(self.mode_transitions)[-3:]])}
        """