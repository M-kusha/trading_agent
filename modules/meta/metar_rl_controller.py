# ─────────────────────────────────────────────────────────────
# File: modules/meta/metar_rl_controller.py
# Enhanced with InfoBus integration & intelligent automation
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import datetime
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from enum import Enum
import json

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit

from modules.meta.ppo_agent import PPOAgent
from modules.meta.ppo_lag_agent import PPOLagAgent


class ControllerMode(Enum):
    """Meta RL Controller operational modes"""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    RETRAINING = "retraining"
    EMERGENCY_STOP = "emergency_stop"
    OPTIMIZATION = "optimization"


class AgentPerformanceTracker:
    """Track and compare agent performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.agent_metrics = defaultdict(lambda: {
            'rewards': deque(maxlen=window_size),
            'losses': deque(maxlen=window_size),
            'episodes': 0,
            'total_reward': 0.0,
            'best_reward': -np.inf,
            'convergence_score': 0.0,
            'stability_score': 0.0,
            'last_update': datetime.datetime.now()
        })
    
    def update_agent_performance(self, agent_name: str, reward: float, loss: float = None):
        """Update performance metrics for an agent"""
        metrics = self.agent_metrics[agent_name]
        
        metrics['rewards'].append(reward)
        if loss is not None:
            metrics['losses'].append(loss)
        
        metrics['episodes'] += 1
        metrics['total_reward'] += reward
        metrics['last_update'] = datetime.datetime.now()
        
        if reward > metrics['best_reward']:
            metrics['best_reward'] = reward
        
        # Calculate convergence score (recent improvement)
        if len(metrics['rewards']) >= 10:
            recent_rewards = list(metrics['rewards'])[-10:]
            older_rewards = list(metrics['rewards'])[-20:-10] if len(metrics['rewards']) >= 20 else []
            
            if older_rewards:
                recent_avg = np.mean(recent_rewards)
                older_avg = np.mean(older_rewards)
                improvement = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
                metrics['convergence_score'] = max(0, min(1, improvement + 0.5))
            
            # Calculate stability score (low variance)
            reward_std = np.std(recent_rewards)
            metrics['stability_score'] = max(0, 1 - reward_std / 100.0)
    
    def get_best_agent(self) -> str:
        """Get the name of the best performing agent"""
        if not self.agent_metrics:
            return "ppo"
        
        best_agent = None
        best_score = -np.inf
        
        for agent_name, metrics in self.agent_metrics.items():
            if metrics['episodes'] < 10:  # Need minimum episodes
                continue
            
            # Combined score: avg reward + convergence + stability
            avg_reward = metrics['total_reward'] / metrics['episodes']
            combined_score = (0.5 * avg_reward + 
                            0.3 * metrics['convergence_score'] * 100 + 
                            0.2 * metrics['stability_score'] * 100)
            
            if combined_score > best_score:
                best_score = combined_score
                best_agent = agent_name
        
        return best_agent or "ppo"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all agents"""
        summary = {}
        
        for agent_name, metrics in self.agent_metrics.items():
            if metrics['episodes'] > 0:
                avg_reward = metrics['total_reward'] / metrics['episodes']
                recent_rewards = list(metrics['rewards'])[-10:] if metrics['rewards'] else [0]
                
                summary[agent_name] = {
                    'episodes': metrics['episodes'],
                    'avg_reward': avg_reward,
                    'best_reward': metrics['best_reward'],
                    'recent_avg': np.mean(recent_rewards),
                    'convergence_score': metrics['convergence_score'],
                    'stability_score': metrics['stability_score'],
                    'last_update': metrics['last_update'].isoformat()
                }
        
        return summary


class MetaRLController(Module, AnalysisMixin, TradingMixin):
    """
    Enhanced meta RL controller with InfoBus integration and intelligent automation.
    Manages multiple RL agents with automatic training, validation, and switching.
    Provides intelligent automation for the entire RL lifecycle.
    """
    
    def __init__(self, obs_size: int, act_size: int = 2, method: str = "ppo-lag", 
                 device: str = "cpu", debug: bool = True, 
                 profit_target: float = 150.0,
                 training_episodes: int = 1000,
                 validation_episodes: int = 100,
                 **kwargs):
        
        # Enhanced configuration
        config = ModuleConfig(
            debug=debug,
            max_history=1000,
            health_check_interval=180,  # 3 minutes
            performance_window=200,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.device = torch.device(device)
        self.obs_size = obs_size
        self.act_size = act_size
        self.profit_target = profit_target
        self.training_episodes = training_episodes
        self.validation_episodes = validation_episodes
        
        # Controller state
        self.current_mode = ControllerMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        self.mode_transitions = deque(maxlen=100)
        
        # Initialize agents
        self._agents = {}
        self._initialize_agents(method)
        
        # Performance tracking
        self.performance_tracker = AgentPerformanceTracker(window_size=100)
        self.training_history = deque(maxlen=1000)
        self.validation_results = deque(maxlen=50)
        
        # Automation configuration
        self.automation_config = {
            'min_training_episodes': 50,
            'convergence_threshold': 0.8,
            'validation_success_rate': 0.6,
            'live_trading_confidence': 0.7,
            'retraining_trigger_loss': -50.0,
            'emergency_stop_loss': -100.0,
            'optimization_interval': 500,  # episodes
            'agent_switch_threshold': 0.2,  # performance difference
            'training_timeout': 3600,  # 1 hour
            'validation_timeout': 600   # 10 minutes
        }
        
        # Training and validation state
        self.current_episode = 0
        self.training_start_time = None
        self.validation_start_time = None
        self.best_validation_score = -np.inf
        self.consecutive_poor_episodes = 0
        self.training_convergence_count = 0
        
        # Live trading state
        self.live_trading_start_time = None
        self.live_session_pnl = 0.0
        self.live_session_trades = 0
        self.live_performance_history = deque(maxlen=100)
        
        # Agent management
        self.active_agent_name = method
        self.active_agent = self._agents[self.active_agent_name]
        self.agent_comparison_results = {}
        
        # Automation intelligence
        self.decision_history = deque(maxlen=200)
        self.automation_metrics = {
            'total_training_sessions': 0,
            'successful_validations': 0,
            'live_trading_sessions': 0,
            'emergency_stops': 0,
            'agent_switches': 0,
            'automation_accuracy': 0.0,
            'avg_training_duration': 0.0,
            'avg_validation_score': 0.0
        }
        
        # Enhanced logging with rotation
        self.logger = RotatingLogger(
            "MetaRLController",
            "logs/strategy/controller/metarl_controller.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("MetaRLController")
        
        self.log_operator_info(
            "🎯 Enhanced Meta RL Controller initialized",
            obs_size=obs_size,
            act_size=act_size,
            active_agent=self.active_agent_name,
            profit_target=f"€{profit_target}",
            training_episodes=training_episodes,
            mode=self.current_mode.value,
            automation_enabled=True
        )
    
    def _initialize_agents(self, preferred_method: str):
        """Initialize all available agents"""
        
        try:
            # Initialize PPO variants
            self._agents = {
                "ppo": PPOAgent(
                    self.obs_size, 
                    act_size=self.act_size, 
                    device=self.device.type, 
                    debug=self.config.debug
                ),
                "ppo-lag": PPOLagAgent(
                    self.obs_size, 
                    act_size=self.act_size, 
                    device=self.device.type, 
                    debug=self.config.debug
                )
            }
            
            # Set active agent
            if preferred_method in self._agents:
                self.active_agent_name = preferred_method
            else:
                self.active_agent_name = "ppo-lag"
                self.log_operator_warning(f"Unknown method {preferred_method}, defaulting to ppo-lag")
            
            self.active_agent = self._agents[self.active_agent_name]
            
            self.log_operator_info(
                "Agents initialized successfully",
                agents=list(self._agents.keys()),
                active_agent=self.active_agent_name
            )
            
        except Exception as e:
            self.log_operator_error(f"Agent initialization failed: {e}")
            raise
    
    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_trading_state()
        
        # Reset all agents
        for agent in self._agents.values():
            agent.reset()
        
        # Reset controller state
        self.current_mode = ControllerMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        self.mode_transitions.clear()
        
        # Reset tracking
        self.performance_tracker = AgentPerformanceTracker(window_size=100)
        self.training_history.clear()
        self.validation_results.clear()
        
        # Reset training state
        self.current_episode = 0
        self.training_start_time = None
        self.validation_start_time = None
        self.best_validation_score = -np.inf
        self.consecutive_poor_episodes = 0
        self.training_convergence_count = 0
        
        # Reset live trading state
        self.live_trading_start_time = None
        self.live_session_pnl = 0.0
        self.live_session_trades = 0
        self.live_performance_history.clear()
        
        # Reset automation state
        self.decision_history.clear()
        
        self.log_operator_info("🔄 Meta RL Controller reset - all state cleared")
    
    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and intelligent automation"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            return
        
        # Extract context and performance data
        context = extract_standard_context(info_bus)
        self._update_performance_from_info_bus(info_bus)
        
        # Execute current mode logic
        self._execute_current_mode(info_bus, context)
        
        # Evaluate mode transitions
        self._evaluate_mode_transition(info_bus, context)
        
        # Update automation metrics
        self._update_automation_metrics()
        
        # Publish controller status
        self._publish_controller_status(info_bus)
    
    def _update_performance_from_info_bus(self, info_bus: InfoBus):
        """Extract and update performance metrics from InfoBus"""
        
        # Extract recent trades and performance
        recent_trades = info_bus.get('recent_trades', [])
        total_pnl = sum(trade.get('pnl', 0) for trade in recent_trades)
        
        if self.current_mode == ControllerMode.LIVE_TRADING:
            self.live_session_pnl += total_pnl
            self.live_session_trades += len(recent_trades)
            
            if recent_trades:
                self.live_performance_history.extend([
                    trade.get('pnl', 0) for trade in recent_trades
                ])
        
        # Extract training metrics
        training_metrics = info_bus.get('training_metrics', {})
        if training_metrics and self.current_mode in [ControllerMode.TRAINING, ControllerMode.RETRAINING]:
            self._update_training_metrics(training_metrics)
        
        # Update trading metrics via mixin
        if total_pnl != 0:
            self._update_trading_metrics({'pnl': total_pnl})
    
    def _update_training_metrics(self, training_metrics: Dict[str, Any]):
        """Update training-specific metrics"""
        
        episode_reward = training_metrics.get('episode_reward_mean', 0)
        episode_loss = training_metrics.get('loss', None)
        self.current_episode = training_metrics.get('episodes', self.current_episode)
        
        # Update performance tracker
        self.performance_tracker.update_agent_performance(
            self.active_agent_name, episode_reward, episode_loss
        )
        
        # Track training history
        training_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'episode': self.current_episode,
            'agent': self.active_agent_name,
            'reward': episode_reward,
            'loss': episode_loss,
            'mode': self.current_mode.value
        }
        self.training_history.append(training_record)
        
        # Check for convergence
        agent_metrics = self.performance_tracker.agent_metrics[self.active_agent_name]
        if agent_metrics['convergence_score'] > self.automation_config['convergence_threshold']:
            self.training_convergence_count += 1
        else:
            self.training_convergence_count = max(0, self.training_convergence_count - 1)
        
        # Track poor performance
        if episode_reward < -10:  # Poor episode
            self.consecutive_poor_episodes += 1
        else:
            self.consecutive_poor_episodes = 0
    
    def _execute_current_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute logic for current controller mode"""
        
        if self.current_mode == ControllerMode.INITIALIZATION:
            self._execute_initialization_mode(info_bus, context)
        elif self.current_mode == ControllerMode.TRAINING:
            self._execute_training_mode(info_bus, context)
        elif self.current_mode == ControllerMode.VALIDATION:
            self._execute_validation_mode(info_bus, context)
        elif self.current_mode == ControllerMode.LIVE_TRADING:
            self._execute_live_trading_mode(info_bus, context)
        elif self.current_mode == ControllerMode.RETRAINING:
            self._execute_retraining_mode(info_bus, context)
        elif self.current_mode == ControllerMode.OPTIMIZATION:
            self._execute_optimization_mode(info_bus, context)
        elif self.current_mode == ControllerMode.EMERGENCY_STOP:
            self._execute_emergency_stop_mode(info_bus, context)
    
    def _execute_initialization_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute initialization mode logic"""
        
        # Perform system health checks
        system_health = self._assess_system_health(info_bus, context)
        
        # Initialize agents if needed
        self._ensure_agents_ready()
        
        # Basic validation of setup
        if system_health > 0.7 and all(agent is not None for agent in self._agents.values()):
            self.log_operator_info(
                "✅ Initialization complete",
                system_health=f"{system_health:.3f}",
                agents_ready=len(self._agents),
                ready_for_training=True
            )
        else:
            self.log_operator_warning(
                "⚠️ Initialization issues detected",
                system_health=f"{system_health:.3f}",
                agents_ready=sum(1 for a in self._agents.values() if a is not None)
            )
    
    def _execute_training_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute training mode logic"""
        
        if not self.training_start_time:
            self.training_start_time = datetime.datetime.now()
            self.log_operator_info(
                "🎓 Training session started",
                agent=self.active_agent_name,
                target_episodes=self.training_episodes
            )
        
        # Monitor training progress
        training_duration = (datetime.datetime.now() - self.training_start_time).total_seconds()
        
        # Log progress periodically
        if self.current_episode % 50 == 0 and self.current_episode > 0:
            agent_metrics = self.performance_tracker.agent_metrics[self.active_agent_name]
            avg_reward = agent_metrics['total_reward'] / max(agent_metrics['episodes'], 1)
            
            self.log_operator_info(
                f"📊 Training progress",
                episode=self.current_episode,
                avg_reward=f"{avg_reward:.3f}",
                convergence_score=f"{agent_metrics['convergence_score']:.3f}",
                duration=f"{training_duration/60:.1f}min"
            )
        
        # Check for training issues
        if self.consecutive_poor_episodes > 20:
            self.log_operator_warning(
                "⚠️ Training performance issues",
                consecutive_poor=self.consecutive_poor_episodes,
                current_episode=self.current_episode
            )
    
    def _execute_validation_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute validation mode logic"""
        
        if not self.validation_start_time:
            self.validation_start_time = datetime.datetime.now()
            self.log_operator_info(
                "🔍 Validation session started",
                agent=self.active_agent_name,
                target_episodes=self.validation_episodes
            )
        
        # Monitor validation progress
        validation_duration = (datetime.datetime.now() - self.validation_start_time).total_seconds()
        
        # Extract validation performance
        recent_performance = self._extract_recent_performance(info_bus)
        
        # Store validation results
        if len(self.validation_results) == 0 or \
           (datetime.datetime.now() - datetime.datetime.fromisoformat(
               self.validation_results[-1]['timestamp'])).total_seconds() > 60:
            
            validation_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'agent': self.active_agent_name,
                'performance': recent_performance,
                'duration': validation_duration,
                'episode_count': self.current_episode
            }
            self.validation_results.append(validation_record)
    
    def _execute_live_trading_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute live trading mode logic"""
        
        if not self.live_trading_start_time:
            self.live_trading_start_time = datetime.datetime.now()
            self.live_session_pnl = 0.0
            self.live_session_trades = 0
            self.log_operator_info(
                "💰 Live trading session started",
                agent=self.active_agent_name,
                profit_target=f"€{self.profit_target}"
            )
        
        # Monitor live trading performance
        trading_duration = (datetime.datetime.now() - self.live_trading_start_time).total_seconds()
        
        # Log performance periodically
        if self.live_session_trades > 0 and self.live_session_trades % 10 == 0:
            avg_pnl_per_trade = self.live_session_pnl / self.live_session_trades
            progress_pct = (self.live_session_pnl / self.profit_target) * 100
            
            self.log_operator_info(
                f"💰 Live trading progress",
                session_pnl=f"€{self.live_session_pnl:.2f}",
                trades=self.live_session_trades,
                avg_per_trade=f"€{avg_pnl_per_trade:.2f}",
                progress=f"{progress_pct:.1f}%",
                duration=f"{trading_duration/60:.1f}min"
            )
        
        # Check for risk management triggers
        if self.live_session_pnl <= self.automation_config['emergency_stop_loss']:
            self.log_operator_error(
                "🚨 Emergency stop triggered",
                session_pnl=f"€{self.live_session_pnl:.2f}",
                threshold=f"€{self.automation_config['emergency_stop_loss']:.2f}"
            )
    
    def _execute_retraining_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute retraining mode logic (similar to training but more focused)"""
        
        if not self.training_start_time:
            self.training_start_time = datetime.datetime.now()
            self.log_operator_info(
                "🔄 Retraining session started",
                agent=self.active_agent_name,
                reason="Performance below threshold"
            )
        
        # Retraining uses accelerated learning
        self._execute_training_mode(info_bus, context)
    
    def _execute_optimization_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute optimization mode - compare agents and select best"""
        
        # Run comparative analysis of all agents
        comparison_results = self._compare_agent_performance()
        
        # Select best agent based on current conditions
        best_agent = self._select_optimal_agent(comparison_results, context)
        
        if best_agent != self.active_agent_name:
            self._switch_active_agent(best_agent, "Optimization identified better agent")
        
        self.log_operator_info(
            "⚙️ Optimization completed",
            best_agent=best_agent,
            comparison_results=len(comparison_results),
            agent_switched=best_agent != self.active_agent_name
        )
    
    def _execute_emergency_stop_mode(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Execute emergency stop mode - halt all trading"""
        
        # Log emergency status
        mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        if mode_duration % 300 == 0:  # Every 5 minutes
            self.log_operator_error(
                "🚨 Emergency stop active",
                duration=f"{mode_duration/60:.1f}min",
                session_loss=f"€{self.live_session_pnl:.2f}",
                action_required="Manual intervention needed"
            )
    
    def _evaluate_mode_transition(self, info_bus: InfoBus, context: Dict[str, Any]):
        """Evaluate whether to transition to a different mode"""
        
        mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        transition_decision = None
        
        if self.current_mode == ControllerMode.INITIALIZATION:
            transition_decision = self._evaluate_initialization_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.TRAINING:
            transition_decision = self._evaluate_training_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.VALIDATION:
            transition_decision = self._evaluate_validation_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.LIVE_TRADING:
            transition_decision = self._evaluate_live_trading_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.RETRAINING:
            transition_decision = self._evaluate_retraining_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.OPTIMIZATION:
            transition_decision = self._evaluate_optimization_transition(info_bus, context, mode_duration)
        elif self.current_mode == ControllerMode.EMERGENCY_STOP:
            transition_decision = self._evaluate_emergency_transition(info_bus, context, mode_duration)
        
        if transition_decision:
            self._execute_mode_transition(transition_decision)
    
    def _evaluate_initialization_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from initialization mode"""
        
        if duration > 60:  # After 1 minute
            system_health = self._assess_system_health(info_bus, context)
            
            if system_health > 0.7:
                return {
                    'target_mode': ControllerMode.TRAINING,
                    'reason': 'Initialization complete, starting training',
                    'priority': 'normal'
                }
            else:
                return {
                    'target_mode': ControllerMode.EMERGENCY_STOP,
                    'reason': f'System health too low: {system_health:.3f}',
                    'priority': 'critical'
                }
        
        return None
    
    def _evaluate_training_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from training mode"""
        
        # Emergency conditions
        if (self.consecutive_poor_episodes > 50 or 
            duration > self.automation_config['training_timeout']):
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': f'Training failure: poor_episodes={self.consecutive_poor_episodes}, duration={duration/60:.1f}min',
                'priority': 'critical'
            }
        
        # Success conditions
        if (self.current_episode >= self.automation_config['min_training_episodes'] and
            self.training_convergence_count >= 5):
            
            agent_metrics = self.performance_tracker.agent_metrics[self.active_agent_name]
            if agent_metrics['convergence_score'] > self.automation_config['convergence_threshold']:
                return {
                    'target_mode': ControllerMode.VALIDATION,
                    'reason': f'Training converged: episodes={self.current_episode}, convergence={agent_metrics["convergence_score"]:.3f}',
                    'priority': 'normal'
                }
        
        # Optimization trigger
        if (self.current_episode > 0 and 
            self.current_episode % self.automation_config['optimization_interval'] == 0):
            return {
                'target_mode': ControllerMode.OPTIMIZATION,
                'reason': f'Optimization interval reached: {self.current_episode} episodes',
                'priority': 'low'
            }
        
        return None
    
    def _evaluate_validation_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from validation mode"""
        
        # Timeout protection
        if duration > self.automation_config['validation_timeout']:
            return {
                'target_mode': ControllerMode.TRAINING,
                'reason': 'Validation timeout, returning to training',
                'priority': 'normal'
            }
        
        # Check validation results
        if len(self.validation_results) >= 3:
            recent_results = list(self.validation_results)[-3:]
            avg_performance = np.mean([
                r['performance'].get('win_rate', 0.5) for r in recent_results
            ])
            
            if avg_performance >= self.automation_config['validation_success_rate']:
                return {
                    'target_mode': ControllerMode.LIVE_TRADING,
                    'reason': f'Validation successful: avg_performance={avg_performance:.3f}',
                    'priority': 'normal'
                }
            else:
                return {
                    'target_mode': ControllerMode.RETRAINING,
                    'reason': f'Validation failed: avg_performance={avg_performance:.3f}',
                    'priority': 'high'
                }
        
        return None
    
    def _evaluate_live_trading_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from live trading mode"""
        
        # Emergency stop conditions
        if self.live_session_pnl <= self.automation_config['emergency_stop_loss']:
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': f'Emergency loss threshold: €{self.live_session_pnl:.2f}',
                'priority': 'critical'
            }
        
        # Retraining conditions
        if self.live_session_pnl <= self.automation_config['retraining_trigger_loss']:
            return {
                'target_mode': ControllerMode.RETRAINING,
                'reason': f'Retraining loss threshold: €{self.live_session_pnl:.2f}',
                'priority': 'high'
            }
        
        # Success conditions - profit target reached
        if self.live_session_pnl >= self.profit_target:
            return {
                'target_mode': ControllerMode.OPTIMIZATION,
                'reason': f'Profit target achieved: €{self.live_session_pnl:.2f}',
                'priority': 'normal'
            }
        
        # Daily session management
        if duration > 28800:  # 8 hours
            return {
                'target_mode': ControllerMode.OPTIMIZATION,
                'reason': 'Daily session complete',
                'priority': 'normal'
            }
        
        return None
    
    def _evaluate_retraining_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from retraining mode"""
        
        # Similar to training but with faster convergence requirements
        if (self.current_episode >= self.automation_config['min_training_episodes'] // 2 and
            self.training_convergence_count >= 3):
            
            return {
                'target_mode': ControllerMode.VALIDATION,
                'reason': f'Retraining complete: episodes={self.current_episode}',
                'priority': 'normal'
            }
        
        # Timeout protection (shorter than regular training)
        if duration > self.automation_config['training_timeout'] // 2:
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': 'Retraining timeout',
                'priority': 'critical'
            }
        
        return None
    
    def _evaluate_optimization_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from optimization mode"""
        
        if duration > 300:  # 5 minutes maximum
            # Determine best next mode based on recent performance
            if self.live_session_pnl > 0:
                return {
                    'target_mode': ControllerMode.LIVE_TRADING,
                    'reason': 'Optimization complete, resuming profitable trading',
                    'priority': 'normal'
                }
            else:
                return {
                    'target_mode': ControllerMode.TRAINING,
                    'reason': 'Optimization complete, improvement needed',
                    'priority': 'normal'
                }
        
        return None
    
    def _evaluate_emergency_transition(self, info_bus: InfoBus, context: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from emergency stop mode"""
        
        # Only allow manual override or after significant time
        if duration > 3600:  # 1 hour
            return {
                'target_mode': ControllerMode.TRAINING,
                'reason': 'Emergency cooldown complete, attempting recovery',
                'priority': 'low'
            }
        
        return None
    
    def _execute_mode_transition(self, transition: Dict[str, Any]):
        """Execute a mode transition"""
        
        target_mode = transition['target_mode']
        reason = transition['reason']
        priority = transition['priority']
        
        old_mode = self.current_mode
        mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        # Record transition
        transition_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'from_mode': old_mode.value,
            'to_mode': target_mode.value,
            'reason': reason,
            'priority': priority,
            'duration': mode_duration,
            'session_pnl': self.live_session_pnl,
            'current_episode': self.current_episode
        }
        self.mode_transitions.append(transition_record)
        
        # Execute transition
        self.current_mode = target_mode
        self.mode_start_time = datetime.datetime.now()
        
        # Reset mode-specific state
        if target_mode == ControllerMode.TRAINING:
            self.training_start_time = None
            self.current_episode = 0
            self.consecutive_poor_episodes = 0
            self.training_convergence_count = 0
            self.automation_metrics['total_training_sessions'] += 1
            
        elif target_mode == ControllerMode.VALIDATION:
            self.validation_start_time = None
            
        elif target_mode == ControllerMode.LIVE_TRADING:
            self.live_trading_start_time = None
            self.live_session_pnl = 0.0
            self.live_session_trades = 0
            self.automation_metrics['live_trading_sessions'] += 1
            
        elif target_mode == ControllerMode.EMERGENCY_STOP:
            self.automation_metrics['emergency_stops'] += 1
        
        # Record decision for learning
        self.decision_history.append(transition_record)
        
        self.log_operator_info(
            f"🔄 Mode transition: {old_mode.value} → {target_mode.value}",
            reason=reason,
            priority=priority,
            duration=f"{mode_duration:.0f}s",
            session_pnl=f"€{self.live_session_pnl:.2f}",
            episode=self.current_episode
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # AGENT MANAGEMENT METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def _compare_agent_performance(self) -> Dict[str, Any]:
        """Compare performance of all agents"""
        
        performance_summary = self.performance_tracker.get_performance_summary()
        best_agent = self.performance_tracker.get_best_agent()
        
        comparison = {
            'timestamp': datetime.datetime.now().isoformat(),
            'best_agent': best_agent,
            'agent_rankings': [],
            'performance_summary': performance_summary
        }
        
        # Rank agents by combined score
        agent_scores = []
        for agent_name, metrics in performance_summary.items():
            combined_score = (0.4 * metrics['avg_reward'] + 
                            0.3 * metrics['convergence_score'] * 100 + 
                            0.3 * metrics['stability_score'] * 100)
            agent_scores.append((agent_name, combined_score, metrics))
        
        # Sort by score (descending)
        agent_scores.sort(key=lambda x: x[1], reverse=True)
        comparison['agent_rankings'] = [
            {
                'agent': agent_name,
                'score': score,
                'metrics': metrics
            }
            for agent_name, score, metrics in agent_scores
        ]
        
        self.agent_comparison_results = comparison
        return comparison
    
    def _select_optimal_agent(self, comparison_results: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Select optimal agent based on comparison and context"""
        
        best_agent = comparison_results['best_agent']
        
        # Consider context factors
        market_regime = context.get('regime', 'unknown')
        volatility = context.get('volatility_level', 'medium')
        
        # Agent preferences by market condition
        agent_preferences = {
            ('trending', 'low'): 'ppo',
            ('trending', 'medium'): 'ppo-lag',
            ('trending', 'high'): 'ppo-lag',
            ('volatile', 'extreme'): 'ppo-lag',
            ('ranging', 'low'): 'ppo'
        }
        
        preferred_agent = agent_preferences.get((market_regime, volatility), best_agent)
        
        # Check if switch is warranted
        if preferred_agent != self.active_agent_name:
            current_performance = self.performance_tracker.agent_metrics[self.active_agent_name]
            preferred_performance = self.performance_tracker.agent_metrics.get(preferred_agent, {})
            
            if (preferred_performance and 
                preferred_performance.get('episodes', 0) > 10 and
                (preferred_performance.get('total_reward', 0) / preferred_performance.get('episodes', 1)) >
                (current_performance.get('total_reward', 0) / max(current_performance.get('episodes', 1), 1)) + 
                self.automation_config['agent_switch_threshold']):
                
                return preferred_agent
        
        return best_agent
    
    def _switch_active_agent(self, new_agent_name: str, reason: str):
        """Switch to a different agent"""
        
        if new_agent_name not in self._agents:
            self.log_operator_error(f"Cannot switch to unknown agent: {new_agent_name}")
            return
        
        old_agent = self.active_agent_name
        self.active_agent_name = new_agent_name
        self.active_agent = self._agents[new_agent_name]
        
        # Record agent switch
        switch_record = {
            'timestamp': datetime.datetime.now().isoformat(),
            'from_agent': old_agent,
            'to_agent': new_agent_name,
            'reason': reason,
            'mode': self.current_mode.value
        }
        
        self.automation_metrics['agent_switches'] += 1
        
        self.log_operator_info(
            f"🔄 Agent switch: {old_agent} → {new_agent_name}",
            reason=reason,
            mode=self.current_mode.value,
            total_switches=self.automation_metrics['agent_switches']
        )
    
    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def record_step(self, obs_vec, reward, **market_data):
        """Enhanced step recording with comprehensive validation"""
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.log_operator_error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.log_operator_error("NaN reward, setting to 0")
                reward = 0.0
            
            # Update performance tracking
            self.performance_tracker.update_agent_performance(self.active_agent_name, reward)
            
            # Validate market data
            validated_market_data = {}
            for key, value in market_data.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    self.log_operator_warning(f"NaN in market data {key}, setting to 0")
                    validated_market_data[key] = 0.0
                else:
                    validated_market_data[key] = value
            
            # Record with active agent
            if self.active_agent_name == "ppo-lag":
                self.active_agent.record_step(obs_vec, reward, **validated_market_data)
            else:
                self.active_agent.record_step(obs_vec, reward)
            
            # Update trading metrics
            self._update_trading_metrics({'pnl': reward})
            
            # Log significant events
            if reward > 20:
                self.log_operator_info(
                    f"💰 Large profit recorded",
                    reward=f"€{reward:.2f}",
                    agent=self.active_agent_name,
                    mode=self.current_mode.value
                )
            elif reward < -20:
                self.log_operator_warning(
                    f"📉 Large loss recorded",
                    loss=f"€{reward:.2f}",
                    agent=self.active_agent_name,
                    mode=self.current_mode.value
                )
            
        except Exception as e:
            self.log_operator_error(f"Step recording failed: {e}")
    
    def end_episode(self, *args, **kwargs):
        """Enhanced episode ending with performance tracking"""
        
        try:
            # Record episode completion
            episode_result = self.active_agent.end_episode(*args, **kwargs)
            
            # Update automation metrics
            self._update_automation_metrics()
            
            # Check for mode-specific actions
            if self.current_mode == ControllerMode.VALIDATION:
                if len(self.validation_results) > 0:
                    latest_validation = self.validation_results[-1]
                    self.automation_metrics['avg_validation_score'] = (
                        0.9 * self.automation_metrics['avg_validation_score'] + 
                        0.1 * latest_validation['performance'].get('win_rate', 0.5)
                    )
            
            return episode_result
            
        except Exception as e:
            self.log_operator_error(f"Episode ending failed: {e}")
    
    def act(self, obs_tensor):
        """Get action from active agent with validation"""
        
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.log_operator_error("NaN in action input tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            # Get action from active agent
            action = self.active_agent.select_action(obs_tensor)
            
            # Validate output
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action = np.asarray(action)
            
            if np.isnan(action).any():
                self.log_operator_error(f"NaN in action output: {action}")
                action = np.nan_to_num(action)
            
            # Apply mode-specific constraints
            if self.current_mode == ControllerMode.EMERGENCY_STOP:
                action = np.zeros_like(action)  # No actions during emergency
            elif self.current_mode in [ControllerMode.VALIDATION, ControllerMode.TRAINING]:
                action = np.clip(action, -0.5, 0.5)  # Reduced risk during training/validation
            
            return action
            
        except Exception as e:
            self.log_operator_error(f"Action generation failed: {e}")
            return np.zeros(self.act_size, dtype=np.float32)
    
    def get_observation_components(self):
        """Enhanced observation components with controller state"""
        
        try:
            # Get base observation from active agent
            base_obs = self.active_agent.get_observation_components()
            
            # Add controller state information
            mode_encoding = {
                ControllerMode.INITIALIZATION: 0.0,
                ControllerMode.TRAINING: 0.2,
                ControllerMode.VALIDATION: 0.4,
                ControllerMode.LIVE_TRADING: 0.6,
                ControllerMode.RETRAINING: 0.8,
                ControllerMode.OPTIMIZATION: 1.0,
                ControllerMode.EMERGENCY_STOP: -1.0
            }
            
            controller_components = [
                mode_encoding.get(self.current_mode, 0.0),
                self.live_session_pnl / 100.0,  # Normalized session PnL
                float(self.current_episode) / 1000.0,  # Normalized episode count
                float(len(self.mode_transitions)) / 100.0  # Normalized transition count
            ]
            
            # Combine observations
            combined_obs = np.concatenate([base_obs, controller_components])
            
            # Validate
            if np.any(np.isnan(combined_obs)):
                self.log_operator_error(f"NaN in combined observation: {combined_obs}")
                combined_obs = np.nan_to_num(combined_obs)
            
            return combined_obs.astype(np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)  # Safe fallback
    
    # ═══════════════════════════════════════════════════════════════════
    # REPORTING AND STATUS METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def get_controller_report(self) -> str:
        """Generate comprehensive controller report"""
        
        # Mode status with emoji
        mode_emoji = {
            ControllerMode.INITIALIZATION: "🔧",
            ControllerMode.TRAINING: "🎓",
            ControllerMode.VALIDATION: "🔍",
            ControllerMode.LIVE_TRADING: "💰",
            ControllerMode.RETRAINING: "🔄",
            ControllerMode.OPTIMIZATION: "⚙️",
            ControllerMode.EMERGENCY_STOP: "🚨"
        }
        
        mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        # Performance summary
        performance_summary = self.performance_tracker.get_performance_summary()
        active_agent_perf = performance_summary.get(self.active_agent_name, {})
        
        # Recent transitions
        recent_transitions = list(self.mode_transitions)[-3:]
        
        return f"""
🎯 META RL CONTROLLER
═══════════════════════════════════════
{mode_emoji.get(self.current_mode, '❓')} Mode: {self.current_mode.value.upper()}
⏱️ Duration: {mode_duration/60:.1f} minutes
🤖 Active Agent: {self.active_agent_name}

💰 SESSION PERFORMANCE
• Session PnL: €{self.live_session_pnl:.2f}
• Target: €{self.profit_target:.2f} ({(self.live_session_pnl/self.profit_target)*100:.1f}%)
• Trades: {self.live_session_trades}
• Episode: {self.current_episode}

🤖 AGENT PERFORMANCE ({self.active_agent_name})
• Episodes: {active_agent_perf.get('episodes', 0)}
• Avg Reward: {active_agent_perf.get('avg_reward', 0.0):.3f}
• Best Reward: {active_agent_perf.get('best_reward', 0.0):.3f}
• Convergence: {active_agent_perf.get('convergence_score', 0.0):.3f}
• Stability: {active_agent_perf.get('stability_score', 0.0):.3f}

🔄 AUTOMATION METRICS
• Training Sessions: {self.automation_metrics['total_training_sessions']}
• Live Sessions: {self.automation_metrics['live_trading_sessions']}
• Emergency Stops: {self.automation_metrics['emergency_stops']}
• Agent Switches: {self.automation_metrics['agent_switches']}
• Automation Accuracy: {self.automation_metrics['automation_accuracy']:.1%}

🔧 AVAILABLE AGENTS
{chr(10).join([f"• {name}: {metrics.get('episodes', 0)} episodes, {metrics.get('avg_reward', 0.0):.3f} avg" for name, metrics in performance_summary.items()])}

🔄 RECENT TRANSITIONS
{chr(10).join([f"• {t['from_mode']} → {t['to_mode']}: {t['reason']}" for t in recent_transitions])}

⚙️ AUTOMATION CONFIG
• Training Episodes: {self.training_episodes}
• Validation Episodes: {self.validation_episodes}
• Profit Target: €{self.profit_target}
• Emergency Threshold: €{self.automation_config['emergency_stop_loss']}
• Retrain Threshold: €{self.automation_config['retraining_trigger_loss']}
        """
    
    def force_mode_transition(self, new_mode: str, reason: str = "Manual override") -> bool:
        """Force a mode transition (for external control)"""
        
        try:
            mode_enum = ControllerMode(new_mode)
            transition = {
                'target_mode': mode_enum,
                'reason': f"Manual: {reason}",
                'priority': 'manual'
            }
            self._execute_mode_transition(transition)
            return True
        except ValueError:
            self.log_operator_error(f"Invalid mode: {new_mode}")
            return False
    
    def get_automation_status(self) -> Dict[str, Any]:
        """Get comprehensive automation status"""
        
        return {
            'controller_state': {
                'mode': self.current_mode.value,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds(),
                'active_agent': self.active_agent_name,
                'current_episode': self.current_episode,
                'automation_enabled': True
            },
            'performance_state': {
                'session_pnl': self.live_session_pnl,
                'session_trades': self.live_session_trades,
                'profit_target': self.profit_target,
                'progress_pct': (self.live_session_pnl / self.profit_target) * 100
            },
            'training_state': {
                'convergence_count': self.training_convergence_count,
                'consecutive_poor': self.consecutive_poor_episodes,
                'training_duration': (datetime.datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0
            },
            'agent_performance': self.performance_tracker.get_performance_summary(),
            'automation_metrics': self.automation_metrics.copy(),
            'recent_transitions': list(self.mode_transitions)[-5:],
            'thresholds': {
                'emergency_stop': self.automation_config['emergency_stop_loss'],
                'retraining_trigger': self.automation_config['retraining_trigger_loss'],
                'convergence_threshold': self.automation_config['convergence_threshold']
            }
        }
    
    # ═══════════════════════════════════════════════════════════════════
    # HELPER METHODS
    # ═══════════════════════════════════════════════════════════════════
    
    def _publish_controller_status(self, info_bus: InfoBus):
        """Publish controller status to InfoBus"""
        
        controller_status = {
            'controller_mode': self.current_mode.value,
            'active_agent': self.active_agent_name,
            'session_pnl': self.live_session_pnl,
            'current_episode': self.current_episode,
            'automation_metrics': self.automation_metrics.copy(),
            'agent_performance': self.performance_tracker.get_performance_summary(),
            'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds(),
            'last_transition': self.mode_transitions[-1] if self.mode_transitions else None
        }
        
        InfoBusUpdater.update_controller_status(info_bus, controller_status)
    
    # Legacy compatibility methods
    def set_mode(self, method: str):
        """Legacy method for agent switching"""
        if method in self._agents:
            self._switch_active_agent(method, "Legacy mode switch")
    
    def obs_dim(self):
        """Return observation dimensionality"""
        return self.obs_size + 4  # Base + controller components
    
    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return controller confidence"""
        
        base_confidence = 0.5
        
        # Mode-based confidence
        if self.current_mode == ControllerMode.LIVE_TRADING:
            if self.live_session_pnl > 0:
                base_confidence += 0.3
        elif self.current_mode == ControllerMode.EMERGENCY_STOP:
            base_confidence = 0.1
        
        # Agent performance confidence
        agent_metrics = self.performance_tracker.agent_metrics[self.active_agent_name]
        if agent_metrics['episodes'] > 10:
            convergence_conf = agent_metrics['convergence_score'] * 0.2
            stability_conf = agent_metrics['stability_score'] * 0.2
            base_confidence += convergence_conf + stability_conf
        
        return float(np.clip(base_confidence, 0.1, 1.0))