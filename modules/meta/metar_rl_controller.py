# ─────────────────────────────────────────────────────────────
# File: modules/meta/meta_rl_controller.py
# 🚀 PRODUCTION-READY Meta RL Controller System
# Enhanced with SmartInfoBus integration & intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker

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


@dataclass
class ControllerConfig:
    """Configuration for Meta RL Controller"""
    obs_size: int = 64
    act_size: int = 2
    method: str = "ppo-lag"
    device: str = "cpu"
    profit_target: float = 150.0
    training_episodes: int = 1000
    validation_episodes: int = 100
    
    # Performance thresholds
    max_processing_time_ms: float = 250
    circuit_breaker_threshold: int = 3
    min_convergence_threshold: float = 0.8
    
    # Automation parameters
    min_training_episodes: int = 50
    validation_success_rate: float = 0.6
    live_trading_confidence: float = 0.7
    retraining_trigger_loss: float = -50.0
    emergency_stop_loss: float = -100.0
    optimization_interval: int = 500


class AgentPerformanceTracker:
    """Track and compare agent performance"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.agent_metrics: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'rewards': deque(maxlen=window_size),
            'losses': deque(maxlen=window_size),
            'episodes': 0,
            'total_reward': 0.0,
            'best_reward': -np.inf,
            'convergence_score': 0.0,
            'stability_score': 0.0,
            'last_update': datetime.now()
        })
    
    def update_agent_performance(self, agent_name: str, reward: float, loss: Optional[float] = None):
        """Update performance metrics for an agent"""
        metrics = self.agent_metrics[agent_name]
        
        metrics['rewards'].append(reward)
        if loss is not None:
            metrics['losses'].append(loss)
        
        metrics['episodes'] += 1
        metrics['total_reward'] += reward
        metrics['last_update'] = datetime.now()
        
        if reward > metrics['best_reward']:
            metrics['best_reward'] = reward
        
        # Calculate convergence score
        if len(metrics['rewards']) >= 10:
            recent_rewards = list(metrics['rewards'])[-10:]
            older_rewards = list(metrics['rewards'])[-20:-10] if len(metrics['rewards']) >= 20 else []
            
            if older_rewards:
                recent_avg = np.mean(recent_rewards)
                older_avg = np.mean(older_rewards)
                improvement = (recent_avg - older_avg) / (abs(older_avg) + 1e-8)
                metrics['convergence_score'] = max(0, min(1, improvement + 0.5))
            
            # Calculate stability score
            reward_std = np.std(recent_rewards)
            metrics['stability_score'] = max(0, 1 - reward_std / 100.0)
    
    def get_best_agent(self) -> str:
        """Get the name of the best performing agent"""
        if not self.agent_metrics:
            return "ppo"
        
        best_agent = None
        best_score = -np.inf
        
        for agent_name, metrics in self.agent_metrics.items():
            if metrics['episodes'] < 10:
                continue
            
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


@module(
    name="MetaRLController",
    version="3.0.0",
    category="meta",
    provides=["controller_status", "agent_performance", "training_metrics", "automation_status"],
    requires=["trades", "actions", "market_data", "training_data"],
    description="Advanced meta RL controller with intelligent automation and agent management",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MetaRLController(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced meta RL controller with SmartInfoBus integration.
    Manages multiple RL agents with automatic training, validation, and switching.
    """

    def __init__(self, 
                 config: Optional[ControllerConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or ControllerConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize controller state
        self._initialize_controller_state()
        
        self.logger.info(
            format_operator_message(
                "🎯", "META_RL_CONTROLLER_INITIALIZED",
                details=f"Obs size: {self.config.obs_size}, Method: {self.config.method}",
                result="Meta RL automation ready",
                context="meta_rl_control"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for meta RL control"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MetaRLController", 
            log_path="logs/meta_rl_controller.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MetaRLController", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for controller operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "obs_size": int(genome.get("obs_size", self.config.obs_size)),
                "act_size": int(genome.get("act_size", self.config.act_size)),
                "method": str(genome.get("method", self.config.method)),
                "profit_target": float(genome.get("profit_target", self.config.profit_target)),
                "training_episodes": int(genome.get("training_episodes", self.config.training_episodes)),
                "validation_episodes": int(genome.get("validation_episodes", self.config.validation_episodes))
            }
        else:
            self.genome = {
                "obs_size": self.config.obs_size,
                "act_size": self.config.act_size,
                "method": self.config.method,
                "profit_target": self.config.profit_target,
                "training_episodes": self.config.training_episodes,
                "validation_episodes": self.config.validation_episodes
            }

    def _initialize_controller_state(self):
        """Initialize meta RL controller state"""
        # Controller state
        self.current_mode = ControllerMode.INITIALIZATION
        self.mode_start_time = datetime.now()
        self.mode_transitions = deque(maxlen=100)
        
        # Initialize agents
        self._agents = {}
        self._initialize_agents(self.genome["method"])
        
        # Performance tracking
        self.performance_tracker_agent = AgentPerformanceTracker(window_size=100)
        self.training_history = deque(maxlen=1000)
        self.validation_results = deque(maxlen=50)
        
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
        self.active_agent_name = self.genome["method"]
        self.active_agent = self._agents[self.active_agent_name]
        self.agent_comparison_results = {}
        
        # Automation metrics
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
        
        # Decision history for learning
        self.decision_history = deque(maxlen=200)

    def _initialize_agents(self, preferred_method: str):
        """Initialize all available agents"""
        try:
            # Initialize PPO variants with fallback handling
            self._agents = {}
            
            # Try to initialize PPO agent
            try:
                self._agents["ppo"] = PPOAgent()
            except Exception as e:
                self.logger.warning(f"Failed to initialize PPOAgent: {e}")
                self._agents["ppo"] = None
            
            # Try to initialize PPO-Lag agent  
            try:
                from modules.meta.ppo_lag_agent import PPOLagAgent
                self._agents["ppo-lag"] = PPOLagAgent()
            except Exception as e:
                self.logger.warning(f"Failed to initialize PPOLagAgent: {e}")
                self._agents["ppo-lag"] = None
            
            # Set active agent
            if preferred_method in self._agents:
                self.active_agent_name = preferred_method
            else:
                self.active_agent_name = "ppo-lag"
                self.logger.warning(f"Unknown method {preferred_method}, defaulting to ppo-lag")
            
            self.active_agent = self._agents[self.active_agent_name]
            
            self.logger.info(
                format_operator_message(
                    "🤖", "AGENTS_INITIALIZED",
                    agents=list(self._agents.keys()),
                    active_agent=self.active_agent_name,
                    context="agent_management"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Agent initialization failed: {e}")
            raise

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_controller_health()
                    self._analyze_controller_performance()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial controller status in SmartInfoBus
            initial_status = {
                "controller_mode": self.current_mode.value,
                "active_agent": self.active_agent_name,
                "session_pnl": 0.0,
                "current_episode": 0
            }
            
            self.smart_bus.set(
                'controller_status',
                initial_status,
                module='MetaRLController',
                thesis="Initial meta RL controller status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process meta RL controller operations"""
        start_time = time.time()
        
        try:
            # Extract controller data
            controller_data = await self._extract_controller_data(**inputs)
            
            if not controller_data:
                return await self._handle_no_data_fallback()
            
            # Update performance metrics
            performance_result = await self._update_performance_metrics(controller_data)
            
            # Execute current mode logic
            mode_result = await self._execute_current_mode(controller_data)
            performance_result.update(mode_result)
            
            # Evaluate mode transitions
            transition_result = await self._evaluate_mode_transition(controller_data)
            performance_result.update(transition_result)
            
            # Update automation metrics
            automation_result = await self._update_automation_metrics()
            performance_result.update(automation_result)
            
            # Generate thesis
            thesis = await self._generate_controller_thesis(controller_data, performance_result)
            
            # Update SmartInfoBus
            await self._update_controller_smart_bus(performance_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return performance_result
            
        except Exception as e:
            return await self._handle_controller_error(e, start_time)

    async def _extract_controller_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract controller data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'MetaRLController') or []
            
            # Get actions
            actions = self.smart_bus.get('actions', 'MetaRLController') or []
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'MetaRLController') or {}
            
            # Get training data
            training_data = self.smart_bus.get('training_data', 'MetaRLController') or {}
            
            # Extract context from market data
            context = self._extract_standard_context(market_data)
            
            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'training_data': training_data,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'obs_vec': inputs.get('obs_vec', None),
                'reward': inputs.get('reward', None)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract controller data: {e}")
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

    async def _update_performance_metrics(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update performance metrics from controller data"""
        try:
            # Extract recent trades and performance
            trades = controller_data.get('trades', [])
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            
            if self.current_mode == ControllerMode.LIVE_TRADING:
                self.live_session_pnl += total_pnl
                self.live_session_trades += len(trades)
                
                if trades:
                    self.live_performance_history.extend([
                        trade.get('pnl', 0) for trade in trades
                    ])
            
            # Extract training metrics
            training_data = controller_data.get('training_data', {})
            if training_data and self.current_mode in [ControllerMode.TRAINING, ControllerMode.RETRAINING]:
                await self._update_training_metrics(training_data)
            
            # Update trading metrics via mixin
            if total_pnl != 0:
                self._update_trading_metrics({'pnl': total_pnl})
            
            return {
                'performance_updated': True,
                'total_pnl': total_pnl,
                'session_pnl': self.live_session_pnl,
                'session_trades': self.live_session_trades
            }
            
        except Exception as e:
            self.logger.error(f"Performance metrics update failed: {e}")
            return {'performance_updated': False, 'error': str(e)}

    async def _update_training_metrics(self, training_data: Dict[str, Any]):
        """Update training-specific metrics"""
        try:
            episode_reward = training_data.get('episode_reward_mean', 0)
            episode_loss = training_data.get('loss', None)
            self.current_episode = training_data.get('episodes', self.current_episode)
            
            # Update performance tracker
            self.performance_tracker_agent.update_agent_performance(
                self.active_agent_name, episode_reward, episode_loss
            )
            
            # Track training history
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'episode': self.current_episode,
                'agent': self.active_agent_name,
                'reward': episode_reward,
                'loss': episode_loss,
                'mode': self.current_mode.value
            }
            self.training_history.append(training_record)
            
            # Check for convergence
            agent_metrics = self.performance_tracker_agent.agent_metrics[self.active_agent_name]
            if agent_metrics['convergence_score'] > self.config.min_convergence_threshold:
                self.training_convergence_count += 1
            else:
                self.training_convergence_count = max(0, self.training_convergence_count - 1)
            
            # Track poor performance
            if episode_reward < -10:
                self.consecutive_poor_episodes += 1
            else:
                self.consecutive_poor_episodes = 0
                
        except Exception as e:
            self.logger.error(f"Training metrics update failed: {e}")

    async def _execute_current_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute logic for current controller mode"""
        try:
            if self.current_mode == ControllerMode.INITIALIZATION:
                return await self._execute_initialization_mode(controller_data)
            elif self.current_mode == ControllerMode.TRAINING:
                return await self._execute_training_mode(controller_data)
            elif self.current_mode == ControllerMode.VALIDATION:
                return await self._execute_validation_mode(controller_data)
            elif self.current_mode == ControllerMode.LIVE_TRADING:
                return await self._execute_live_trading_mode(controller_data)
            elif self.current_mode == ControllerMode.RETRAINING:
                return await self._execute_retraining_mode(controller_data)
            elif self.current_mode == ControllerMode.OPTIMIZATION:
                return await self._execute_optimization_mode(controller_data)
            elif self.current_mode == ControllerMode.EMERGENCY_STOP:
                return await self._execute_emergency_stop_mode(controller_data)
            else:
                return {'mode_executed': False, 'reason': 'unknown_mode'}
                
        except Exception as e:
            self.logger.error(f"Mode execution failed: {e}")
            return self._create_fallback_response("mode execution failed")

    async def _execute_initialization_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute initialization mode logic"""
        try:
            # Perform system health checks
            system_health = self._assess_system_health(controller_data)
            
            # Ensure agents are ready
            agents_ready = all(agent is not None for agent in self._agents.values())
            
            return {
                'mode_executed': True,
                'system_health': system_health,
                'agents_ready': agents_ready,
                'ready_for_training': system_health > 0.7 and agents_ready
            }
            
        except Exception as e:
            self.logger.error(f"Initialization mode failed: {e}")
            return self._create_fallback_response("initialization mode failed")

    async def _execute_training_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute training mode logic"""
        try:
            if not self.training_start_time:
                self.training_start_time = datetime.now()
            
            training_duration = (datetime.now() - self.training_start_time).total_seconds()
            
            # Process training step if data available
            obs_vec = controller_data.get('obs_vec')
            reward = controller_data.get('reward')
            
            if obs_vec is not None and reward is not None:
                await self._process_training_step(obs_vec, reward, controller_data)
            
            return {
                'mode_executed': True,
                'training_active': True,
                'current_episode': self.current_episode,
                'training_duration': training_duration,
                'convergence_count': self.training_convergence_count,
                'poor_episodes': self.consecutive_poor_episodes
            }
            
        except Exception as e:
            self.logger.error(f"Training mode failed: {e}")
            return self._create_fallback_response("training mode failed")

    async def _execute_validation_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation mode logic"""
        try:
            if not self.validation_start_time:
                self.validation_start_time = datetime.now()
            
            validation_duration = (datetime.now() - self.validation_start_time).total_seconds()
            
            # Extract validation performance
            recent_performance = self._extract_recent_performance(controller_data)
            
            # Store validation results
            if len(self.validation_results) == 0 or \
               (datetime.now() - datetime.fromisoformat(
                   self.validation_results[-1]['timestamp'])).total_seconds() > 60:
                
                validation_record = {
                    'timestamp': datetime.now().isoformat(),
                    'agent': self.active_agent_name,
                    'performance': recent_performance,
                    'duration': validation_duration,
                    'episode_count': self.current_episode
                }
                self.validation_results.append(validation_record)
            
            return {
                'mode_executed': True,
                'validation_active': True,
                'validation_duration': validation_duration,
                'performance': recent_performance,
                'validation_results': len(self.validation_results)
            }
            
        except Exception as e:
            self.logger.error(f"Validation mode failed: {e}")
            return self._create_fallback_response("validation mode failed")

    async def _execute_live_trading_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute live trading mode logic"""
        try:
            if not self.live_trading_start_time:
                self.live_trading_start_time = datetime.now()
                self.live_session_pnl = 0.0
                self.live_session_trades = 0
            
            trading_duration = (datetime.now() - self.live_trading_start_time).total_seconds()
            
            # Calculate progress towards target
            progress_pct = (self.live_session_pnl / self.genome["profit_target"]) * 100
            
            return {
                'mode_executed': True,
                'live_trading_active': True,
                'session_pnl': self.live_session_pnl,
                'session_trades': self.live_session_trades,
                'trading_duration': trading_duration,
                'progress_pct': progress_pct,
                'profit_target': self.genome["profit_target"]
            }
            
        except Exception as e:
            self.logger.error(f"Live trading mode failed: {e}")
            return self._create_fallback_response("live trading mode failed")

    async def _execute_retraining_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retraining mode logic"""
        try:
            # Similar to training but more focused
            if not self.training_start_time:
                self.training_start_time = datetime.now()
            
            return await self._execute_training_mode(controller_data)
            
        except Exception as e:
            self.logger.error(f"Retraining mode failed: {e}")
            return self._create_fallback_response("retraining mode failed")

    async def _execute_optimization_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization mode - compare agents and select best"""
        try:
            # Run comparative analysis of all agents
            comparison_results = self._compare_agent_performance()
            
            # Select best agent based on current conditions
            best_agent = self._select_optimal_agent(comparison_results, controller_data['context'])
            
            agent_switched = False
            if best_agent != self.active_agent_name:
                self._switch_active_agent(best_agent, "Optimization identified better agent")
                agent_switched = True
            
            return {
                'mode_executed': True,
                'optimization_completed': True,
                'best_agent': best_agent,
                'agent_switched': agent_switched,
                'comparison_results': len(comparison_results.get('agent_rankings', []))
            }
            
        except Exception as e:
            self.logger.error(f"Optimization mode failed: {e}")
            return self._create_fallback_response("optimization mode failed")

    async def _execute_emergency_stop_mode(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute emergency stop mode - halt all trading"""
        try:
            mode_duration = (datetime.now() - self.mode_start_time).total_seconds()
            
            return {
                'mode_executed': True,
                'emergency_active': True,
                'emergency_duration': mode_duration,
                'session_loss': self.live_session_pnl,
                'requires_intervention': True
            }
            
        except Exception as e:
            self.logger.error(f"Emergency stop mode failed: {e}")
            return self._create_fallback_response("emergency stop mode failed")

    async def _evaluate_mode_transition(self, controller_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate whether to transition to a different mode"""
        try:
            mode_duration = (datetime.now() - self.mode_start_time).total_seconds()
            transition_decision = None
            
            if self.current_mode == ControllerMode.INITIALIZATION:
                transition_decision = self._evaluate_initialization_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.TRAINING:
                transition_decision = self._evaluate_training_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.VALIDATION:
                transition_decision = self._evaluate_validation_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.LIVE_TRADING:
                transition_decision = self._evaluate_live_trading_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.RETRAINING:
                transition_decision = self._evaluate_retraining_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.OPTIMIZATION:
                transition_decision = self._evaluate_optimization_transition(controller_data, mode_duration)
            elif self.current_mode == ControllerMode.EMERGENCY_STOP:
                transition_decision = self._evaluate_emergency_transition(controller_data, mode_duration)
            
            if transition_decision:
                await self._execute_mode_transition(transition_decision)
                return {
                    'mode_transitioned': True,
                    'new_mode': self.current_mode.value,
                    'transition_reason': transition_decision['reason']
                }
            
            return {
                'mode_transitioned': False,
                'current_mode': self.current_mode.value,
                'mode_duration': mode_duration
            }
            
        except Exception as e:
            self.logger.error(f"Mode transition evaluation failed: {e}")
            return {'mode_transitioned': False, 'error': str(e)}

    def _evaluate_initialization_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from initialization mode"""
        if duration > 60:  # After 1 minute
            system_health = self._assess_system_health(controller_data)
            
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

    def _evaluate_training_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from training mode"""
        # Emergency conditions
        if (self.consecutive_poor_episodes > 50 or duration > 3600):  # 1 hour timeout
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': f'Training failure: poor_episodes={self.consecutive_poor_episodes}',
                'priority': 'critical'
            }
        
        # Success conditions
        if (self.current_episode >= self.config.min_training_episodes and
            self.training_convergence_count >= 5):
            
            agent_metrics = self.performance_tracker_agent.agent_metrics[self.active_agent_name]
            if agent_metrics['convergence_score'] > self.config.min_convergence_threshold:
                return {
                    'target_mode': ControllerMode.VALIDATION,
                    'reason': f'Training converged: episodes={self.current_episode}',
                    'priority': 'normal'
                }
        
        # Optimization trigger
        if (self.current_episode > 0 and 
            self.current_episode % self.config.optimization_interval == 0):
            return {
                'target_mode': ControllerMode.OPTIMIZATION,
                'reason': f'Optimization interval reached: {self.current_episode} episodes',
                'priority': 'low'
            }
        
        return None

    def _evaluate_validation_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from validation mode"""
        # Timeout protection
        if duration > 600:  # 10 minutes
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
            
            if avg_performance >= self.config.validation_success_rate:
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

    def _evaluate_live_trading_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from live trading mode"""
        # Emergency stop conditions
        if self.live_session_pnl <= self.config.emergency_stop_loss:
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': f'Emergency loss threshold: €{self.live_session_pnl:.2f}',
                'priority': 'critical'
            }
        
        # Retraining conditions
        if self.live_session_pnl <= self.config.retraining_trigger_loss:
            return {
                'target_mode': ControllerMode.RETRAINING,
                'reason': f'Retraining loss threshold: €{self.live_session_pnl:.2f}',
                'priority': 'high'
            }
        
        # Success conditions
        if self.live_session_pnl >= self.genome["profit_target"]:
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

    def _evaluate_retraining_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from retraining mode"""
        # Similar to training but faster
        if (self.current_episode >= self.config.min_training_episodes // 2 and
            self.training_convergence_count >= 3):
            
            return {
                'target_mode': ControllerMode.VALIDATION,
                'reason': f'Retraining complete: episodes={self.current_episode}',
                'priority': 'normal'
            }
        
        # Timeout protection
        if duration > 1800:  # 30 minutes
            return {
                'target_mode': ControllerMode.EMERGENCY_STOP,
                'reason': 'Retraining timeout',
                'priority': 'critical'
            }
        
        return None

    def _evaluate_optimization_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
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

    def _evaluate_emergency_transition(self, controller_data: Dict[str, Any], duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from emergency stop mode"""
        # Only allow manual override or after significant time
        if duration > 3600:  # 1 hour
            return {
                'target_mode': ControllerMode.TRAINING,
                'reason': 'Emergency cooldown complete, attempting recovery',
                'priority': 'low'
            }
        
        return None

    async def _execute_mode_transition(self, transition: Dict[str, Any]):
        """Execute a mode transition"""
        try:
            target_mode = transition['target_mode']
            reason = transition['reason']
            priority = transition['priority']
            
            old_mode = self.current_mode
            mode_duration = (datetime.now() - self.mode_start_time).total_seconds()
            
            # Record transition
            transition_record = {
                'timestamp': datetime.now().isoformat(),
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
            self.mode_start_time = datetime.now()
            
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
            
            # Record decision
            self.decision_history.append(transition_record)
            
            self.logger.info(
                format_operator_message(
                    "🔄", "MODE_TRANSITION",
                    from_mode=old_mode.value,
                    to_mode=target_mode.value,
                    reason=reason,
                    priority=priority,
                    context="mode_transition"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Mode transition execution failed: {e}")

    async def _process_training_step(self, obs_vec: np.ndarray, reward: float, controller_data: Dict[str, Any]):
        """Process a training step with the active agent"""
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0
            
            # Record step with active agent
            market_data = controller_data.get('market_data', {})
            if self.active_agent_name == "ppo-lag":
                self.active_agent.record_step(
                    obs_vec, reward,
                    price=market_data.get('price', 0),
                    volume=market_data.get('volume', 0),
                    spread=market_data.get('spread', 0),
                    volatility=market_data.get('volatility', 1)
                )
            else:
                self.active_agent.record_step(obs_vec, reward)
            
        except Exception as e:
            self.logger.error(f"Training step processing failed: {e}")

    def _compare_agent_performance(self) -> Dict[str, Any]:
        """Compare performance of all agents"""
        performance_summary = self.performance_tracker_agent.get_performance_summary()
        best_agent = self.performance_tracker_agent.get_best_agent()
        
        comparison = {
            'timestamp': datetime.now().isoformat(),
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
        if preferred_agent and preferred_agent != self.active_agent_name:
            current_performance = self.performance_tracker_agent.agent_metrics[self.active_agent_name]
            preferred_performance = self.performance_tracker_agent.agent_metrics.get(preferred_agent, {})
            
            if (preferred_performance and 
                preferred_performance.get('episodes', 0) > 10 and
                (preferred_performance.get('total_reward', 0) / max(preferred_performance.get('episodes', 1), 1)) >
                (current_performance.get('total_reward', 0) / max(current_performance.get('episodes', 1), 1)) + 0.2):
                
                return preferred_agent
        
        return best_agent or "ppo"

    def _switch_active_agent(self, new_agent_name: str, reason: str):
        """Switch to a different agent"""
        if new_agent_name not in self._agents:
            self.logger.error(f"Cannot switch to unknown agent: {new_agent_name}")
            return
        
        old_agent = self.active_agent_name
        self.active_agent_name = new_agent_name
        self.active_agent = self._agents[new_agent_name]
        
        self.automation_metrics['agent_switches'] += 1
        
        self.logger.info(
            format_operator_message(
                "🔄", "AGENT_SWITCH",
                from_agent=old_agent,
                to_agent=new_agent_name,
                reason=reason,
                context="agent_management"
            )
        )

    async def _update_automation_metrics(self) -> Dict[str, Any]:
        """Update automation metrics"""
        try:
            # Calculate automation accuracy
            if len(self.decision_history) > 10:
                successful_decisions = sum(1 for d in self.decision_history if d.get('priority') != 'critical')
                self.automation_metrics['automation_accuracy'] = successful_decisions / len(self.decision_history)
            
            # Calculate average training duration
            if self.automation_metrics['total_training_sessions'] > 0:
                total_duration = sum(
                    (datetime.fromisoformat(h['timestamp']) - 
                     datetime.fromisoformat(h['timestamp'])).total_seconds() 
                    for h in self.training_history
                    if h.get('mode') == 'training'
                )
                self.automation_metrics['avg_training_duration'] = total_duration / self.automation_metrics['total_training_sessions']
            
            # Calculate average validation score
            if self.validation_results:
                avg_score = np.mean([
                    r['performance'].get('win_rate', 0.5) for r in self.validation_results
                ])
                self.automation_metrics['avg_validation_score'] = avg_score
            
            return {
                'automation_updated': True,
                'automation_accuracy': self.automation_metrics['automation_accuracy'],
                'total_sessions': self.automation_metrics['total_training_sessions']
            }
            
        except Exception as e:
            self.logger.error(f"Automation metrics update failed: {e}")
            return {'automation_updated': False, 'error': str(e)}

    async def _generate_controller_thesis(self, controller_data: Dict[str, Any], 
                                         controller_result: Dict[str, Any]) -> str:
        """Generate comprehensive controller thesis"""
        try:
            # Controller metrics
            current_mode = self.current_mode.value
            active_agent = self.active_agent_name
            session_pnl = self.live_session_pnl
            
            # Performance metrics
            mode_executed = controller_result.get('mode_executed', False)
            current_episode = self.current_episode
            
            thesis_parts = [
                f"Meta RL Controller: Mode {current_mode} with agent {active_agent}",
                f"Session performance: €{session_pnl:.2f} PnL across {self.live_session_trades} trades"
            ]
            
            # Mode-specific details
            if current_mode == 'training':
                convergence_count = self.training_convergence_count
                poor_episodes = self.consecutive_poor_episodes
                thesis_parts.append(f"Training: Episode {current_episode}, convergence {convergence_count}, poor episodes {poor_episodes}")
            
            elif current_mode == 'live_trading':
                progress_pct = controller_result.get('progress_pct', 0)
                thesis_parts.append(f"Live trading: {progress_pct:.1f}% progress toward €{self.genome['profit_target']:.0f} target")
            
            elif current_mode == 'validation':
                validation_results = len(self.validation_results)
                thesis_parts.append(f"Validation: {validation_results} validation runs completed")
            
            # Automation metrics
            automation_accuracy = self.automation_metrics['automation_accuracy']
            emergency_stops = self.automation_metrics['emergency_stops']
            thesis_parts.append(f"Automation: {automation_accuracy:.1%} accuracy, {emergency_stops} emergency stops")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Controller thesis generation failed: {str(e)} - Meta RL control continuing"

    async def _update_controller_smart_bus(self, controller_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with controller results"""
        try:
            # Controller status
            controller_status = {
                'controller_mode': self.current_mode.value,
                'active_agent': self.active_agent_name,
                'session_pnl': self.live_session_pnl,
                'current_episode': self.current_episode,
                'mode_duration': (datetime.now() - self.mode_start_time).total_seconds(),
                'last_transition': self.mode_transitions[-1] if self.mode_transitions else None
            }
            
            self.smart_bus.set(
                'controller_status',
                controller_status,
                module='MetaRLController',
                thesis=thesis
            )
            
            # Agent performance
            agent_performance = {
                'performance_summary': self.performance_tracker_agent.get_performance_summary(),
                'best_agent': self.performance_tracker_agent.get_best_agent(),
                'agent_comparison': self.agent_comparison_results
            }
            
            self.smart_bus.set(
                'agent_performance',
                agent_performance,
                module='MetaRLController',
                thesis=f"Agent performance: {len(self._agents)} agents tracked"
            )
            
            # Training metrics
            training_metrics = {
                'training_history': list(self.training_history)[-10:],
                'validation_results': list(self.validation_results)[-5:],
                'convergence_count': self.training_convergence_count,
                'poor_episodes': self.consecutive_poor_episodes
            }
            
            self.smart_bus.set(
                'training_metrics',
                training_metrics,
                module='MetaRLController',
                thesis="Training metrics and convergence tracking"
            )
            
            # Automation status
            automation_status = {
                'automation_metrics': self.automation_metrics.copy(),
                'mode_transitions': list(self.mode_transitions)[-5:],
                'decision_history': list(self.decision_history)[-10:]
            }
            
            self.smart_bus.set(
                'automation_status',
                automation_status,
                module='MetaRLController',
                thesis="Automation status and decision tracking"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    def _assess_system_health(self, controller_data: Dict[str, Any]) -> float:
        """Assess overall system health"""
        health_score = 0.7  # Base health
        
        # Check data availability
        if controller_data.get('trades') and controller_data.get('market_data'):
            health_score += 0.2
        
        # Check agent health
        if all(agent is not None for agent in self._agents.values()):
            health_score += 0.1
        
        return min(1.0, health_score)

    def _extract_recent_performance(self, controller_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract recent performance metrics"""
        trades = controller_data.get('trades', [])
        
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

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no controller data is available"""
        self.logger.warning("No controller data available - using cached state")
        
        return {
            'controller_mode': self.current_mode.value,
            'active_agent': self.active_agent_name,
            'session_pnl': self.live_session_pnl,
            'fallback_reason': 'no_controller_data'
        }

    async def _handle_controller_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle controller operation errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MetaRLController")
        explanation = self.english_explainer.explain_error(
            "MetaRLController", str(error), "controller operations"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "CONTROLLER_OPERATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="meta_rl_control"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'controller_mode': self.current_mode.value,
            'active_agent': self.active_agent_name,
            'session_pnl': self.live_session_pnl,
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_controller_health(self):
        """Update controller health metrics"""
        try:
            # Check performance metrics
            if self.current_mode == ControllerMode.LIVE_TRADING:
                if self.live_session_pnl < self.config.retraining_trigger_loss:
                    self._health_status = 'warning'
                elif self.live_session_pnl > 0:
                    self._health_status = 'healthy'
            elif self.current_mode == ControllerMode.EMERGENCY_STOP:
                self._health_status = 'critical'
            else:
                self._health_status = 'healthy'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_controller_performance(self):
        """Analyze controller performance metrics"""
        try:
            # Check automation accuracy
            automation_accuracy = self.automation_metrics['automation_accuracy']
            if automation_accuracy > 0.8:
                self.logger.info(
                    format_operator_message(
                        "🎯", "HIGH_AUTOMATION_ACCURACY",
                        accuracy=f"{automation_accuracy:.1%}",
                        total_decisions=len(self.decision_history),
                        context="automation_performance"
                    )
                )
            
        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MetaRLController', 'controller_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MetaRLController', 'controller_cycle', 0, False
        )

    def calculate_confidence(self, obs: Any = None, **kwargs) -> float:
        """Calculate confidence in controller recommendations"""
        base_confidence = 0.5
        
        # Mode-based confidence
        if self.current_mode == ControllerMode.LIVE_TRADING:
            if self.live_session_pnl > 0:
                base_confidence += 0.3
        elif self.current_mode == ControllerMode.EMERGENCY_STOP:
            base_confidence = 0.1
        
        # Agent performance confidence
        agent_metrics = self.performance_tracker_agent.agent_metrics[self.active_agent_name]
        if agent_metrics['episodes'] > 10:
            convergence_conf = agent_metrics['convergence_score'] * 0.2
            stability_conf = agent_metrics['stability_score'] * 0.2
            base_confidence += convergence_conf + stability_conf
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'config': self.config.__dict__,
            'genome': self.genome.copy(),
            'current_mode': self.current_mode.value,
            'mode_start_time': self.mode_start_time.isoformat(),
            'active_agent_name': self.active_agent_name,
            'current_episode': self.current_episode,
            'live_session_pnl': self.live_session_pnl,
            'live_session_trades': self.live_session_trades,
            'training_convergence_count': self.training_convergence_count,
            'consecutive_poor_episodes': self.consecutive_poor_episodes,
            'automation_metrics': self.automation_metrics.copy(),
            'mode_transitions': list(self.mode_transitions),
            'training_history': list(self.training_history)[-50:],
            'validation_results': list(self.validation_results),
            'health_status': self._health_status,
            'circuit_breaker': self.circuit_breaker.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'current_mode' in state:
            self.current_mode = ControllerMode(state['current_mode'])
        
        if 'mode_start_time' in state:
            self.mode_start_time = datetime.fromisoformat(state['mode_start_time'])
        
        if 'active_agent_name' in state:
            self.active_agent_name = state['active_agent_name']
            if self.active_agent_name in self._agents:
                self.active_agent = self._agents[self.active_agent_name]
        
        if 'current_episode' in state:
            self.current_episode = state['current_episode']
        
        if 'live_session_pnl' in state:
            self.live_session_pnl = state['live_session_pnl']
        
        if 'live_session_trades' in state:
            self.live_session_trades = state['live_session_trades']
        
        if 'training_convergence_count' in state:
            self.training_convergence_count = state['training_convergence_count']
        
        if 'consecutive_poor_episodes' in state:
            self.consecutive_poor_episodes = state['consecutive_poor_episodes']
        
        if 'automation_metrics' in state:
            self.automation_metrics.update(state['automation_metrics'])
        
        if 'mode_transitions' in state:
            self.mode_transitions = deque(state['mode_transitions'], maxlen=100)
        
        if 'training_history' in state:
            self.training_history = deque(state['training_history'], maxlen=1000)
        
        if 'validation_results' in state:
            self.validation_results = deque(state['validation_results'], maxlen=50)
        
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
            'current_mode': self.current_mode.value,
            'active_agent': self.active_agent_name,
            'session_pnl': self.live_session_pnl
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def record_step(self, obs_vec: np.ndarray, reward: float, **market_data):
        """Legacy compatibility for step recording"""
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0
            
            # Update performance tracking
            self.performance_tracker_agent.update_agent_performance(self.active_agent_name, reward)
            
            # Record with active agent
            if self.active_agent_name == "ppo-lag":
                self.active_agent.record_step(obs_vec, reward, **market_data)
            else:
                self.active_agent.record_step(obs_vec, reward)
            
        except Exception as e:
            self.logger.error(f"Step recording failed: {e}")

    def end_episode(self, *args, **kwargs):
        """Legacy compatibility for episode ending"""
        try:
            episode_result = self.active_agent.end_episode(*args, **kwargs)
            return episode_result
        except Exception as e:
            self.logger.error(f"Episode ending failed: {e}")

    def act(self, obs_tensor: torch.Tensor) -> np.ndarray:
        """Legacy compatibility for action selection"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            # Get action from active agent
            action = self.active_agent.select_action(obs_tensor)
            
            # Validate output
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action = np.asarray(action)
            
            if np.isnan(action).any():
                action = np.nan_to_num(action)
            
            # Apply mode-specific constraints
            if self.current_mode == ControllerMode.EMERGENCY_STOP:
                action = np.zeros_like(action)
            elif self.current_mode in [ControllerMode.VALIDATION, ControllerMode.TRAINING]:
                action = np.clip(action, -0.5, 0.5)
            
            return action
            
        except Exception as e:
            self.logger.error(f"Action generation failed: {e}")
            return np.zeros(self.genome["act_size"], dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        """Legacy compatibility for observation components"""
        try:
            # Mode encoding
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
                self.live_session_pnl / 100.0,
                float(self.current_episode) / 1000.0,
                float(len(self.mode_transitions)) / 100.0
            ]
            
            # Get base observation from active agent
            try:
                base_obs = self.active_agent.get_observation_components()
                combined_obs = np.concatenate([base_obs, controller_components])
            except:
                combined_obs = np.array(controller_components, dtype=np.float32)
            
            # Validate
            if np.any(np.isnan(combined_obs)):
                combined_obs = np.nan_to_num(combined_obs)
            
            return combined_obs.astype(np.float32)
            
        except Exception as e:
            return np.zeros(8, dtype=np.float32)

    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        return np.zeros(self.genome["act_size"], dtype=np.float32)

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        return self.calculate_confidence(obs, **kwargs)

    def set_mode(self, method: str):
        """Legacy method for agent switching"""
        if method in self._agents:
            self._switch_active_agent(method, "Legacy mode switch")

    def obs_dim(self) -> int:
        """Return observation dimensionality"""
        return self.genome["obs_size"] + 4

    def force_mode_transition(self, new_mode: str, reason: str = "Manual override") -> bool:
        """Force a mode transition (for external control)"""
        try:
            mode_enum = ControllerMode(new_mode)
            transition = {
                'target_mode': mode_enum,
                'reason': f"Manual: {reason}",
                'priority': 'manual'
            }
            asyncio.create_task(self._execute_mode_transition(transition))
            return True
        except ValueError:
            self.logger.error(f"Invalid mode: {new_mode}")
            return False