# ─────────────────────────────────────────────────────────────
# File: modules/meta/meta_agent.py
# 🚀 PRODUCTION-READY Meta Agent System
# Advanced automation with SmartInfoBus integration and intelligent decision making
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


class MetaMode(Enum):
    """Meta agent operational modes"""
    INITIALIZATION = "initialization"
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading" 
    RETRAINING = "retraining"
    EVALUATION = "evaluation"
    EMERGENCY_STOP = "emergency_stop"
    OPTIMIZATION = "optimization"


@dataclass
class MetaAgentConfig:
    """Configuration for Meta Agent"""
    window: int = 20
    profit_target: float = 150.0
    retrain_threshold: float = -50.0
    emergency_threshold: float = -100.0
    confidence_threshold: float = 0.7
    
    # Automation parameters
    min_training_episodes: int = 100
    convergence_episodes: int = 10
    improvement_threshold: float = 0.05
    live_evaluation_period: int = 3600  # 1 hour
    retrain_cooldown: int = 7200  # 2 hours
    emergency_cooldown: int = 1800  # 30 minutes
    
    # Performance thresholds
    max_processing_time_ms: float = 300
    circuit_breaker_threshold: int = 3
    min_automation_score: float = 0.4
    
    # Decision parameters
    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    decision_history_size: int = 100


@module(
    name="MetaAgent",
    version="3.0.0",
    category="meta",
    provides=["automation_decisions", "system_mode", "automation_metrics", "meta_performance"],
    requires=["system_performance", "training_metrics", "market_conditions", "risk_signals"],
    description="Advanced meta agent for autonomous trading system automation and lifecycle management",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MetaAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced meta agent with SmartInfoBus integration.
    Manages the entire trading lifecycle with intelligent automation decisions.
    """

    def __init__(self, 
                 config: Optional[MetaAgentConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or MetaAgentConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize meta agent state
        self._initialize_meta_state()
        
        self.logger.info(
            format_operator_message(
                "🤖", "META_AGENT_INITIALIZED",
                details=f"Modes: {len(MetaMode)}, Profit target: €{self.config.profit_target}",
                result="Meta agent ready for automation",
                context="meta_initialization"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for meta agent"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MetaAgent", 
            log_path="logs/meta/meta_agent.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MetaAgent", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for automation operations
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
                "profit_target": float(genome.get("profit_target", self.config.profit_target)),
                "retrain_threshold": float(genome.get("retrain_threshold", self.config.retrain_threshold)),
                "emergency_threshold": float(genome.get("emergency_threshold", self.config.emergency_threshold)),
                "confidence_threshold": float(genome.get("confidence_threshold", self.config.confidence_threshold)),
                "min_training_episodes": int(genome.get("min_training_episodes", self.config.min_training_episodes)),
                "improvement_threshold": float(genome.get("improvement_threshold", self.config.improvement_threshold)),
                "confidence_decay": float(genome.get("confidence_decay", self.config.confidence_decay)),
                "performance_smoothing": float(genome.get("performance_smoothing", self.config.performance_smoothing))
            }
            # Update config with genome values
            for key, value in self.genome.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        else:
            self.genome = {
                "profit_target": self.config.profit_target,
                "retrain_threshold": self.config.retrain_threshold,
                "emergency_threshold": self.config.emergency_threshold,
                "confidence_threshold": self.config.confidence_threshold,
                "min_training_episodes": self.config.min_training_episodes,
                "improvement_threshold": self.config.improvement_threshold,
                "confidence_decay": self.config.confidence_decay,
                "performance_smoothing": self.config.performance_smoothing
            }

    def _initialize_meta_state(self):
        """Initialize meta agent state"""
        # Operational state
        self.current_mode = MetaMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        self.mode_transitions = deque(maxlen=50)
        
        # Performance tracking
        self.daily_pnl = 0.0
        self.session_pnl = 0.0
        self.peak_pnl = 0.0
        self.drawdown_pct = 0.0
        self.consecutive_losses = 0
        self.win_streak = 0
        self.loss_streak = 0
        
        # Training performance
        self.training_episodes = 0
        self.training_start_time = None
        self.best_training_reward = -np.inf
        self.training_convergence_count = 0
        self.validation_performance = deque(maxlen=20)
        
        # Automation state
        self.system_confidence = 0.5
        self.automation_score = 0.0
        self.last_decision_time = None
        self.last_retrain_time = None
        self.last_emergency_time = None
        self.evaluation_start_time = None
        
        # Decision tracking
        self.decision_history = deque(maxlen=self.config.decision_history_size)
        self.automation_metrics = {
            'total_mode_switches': 0,
            'successful_live_sessions': 0,
            'emergency_stops': 0,
            'retraining_sessions': 0,
            'avg_training_duration': 0.0,
            'avg_live_duration': 0.0,
            'automation_accuracy': 0.0,
            'decision_confidence': 0.0
        }
        
        # System monitoring
        self._performance_history = deque(maxlen=100)
        self._confidence_history = deque(maxlen=100)
        self._mode_performance = defaultdict(list)
        self._decision_effectiveness = deque(maxlen=50)
        
        # Risk management
        self._risk_alerts = deque(maxlen=20)
        self._system_warnings = deque(maxlen=30)
        self._emergency_triggers = []

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_meta_health()
                    self._analyze_automation_effectiveness()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial meta agent status in SmartInfoBus
            initial_status = {
                "current_mode": self.current_mode.value,
                "system_confidence": self.system_confidence,
                "automation_score": 0.0,
                "total_mode_switches": 0
            }
            
            self.smart_bus.set(
                'automation_decisions',
                initial_status,
                module='MetaAgent',
                thesis="Initial meta agent automation status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process meta agent automation"""
        start_time = time.time()
        
        try:
            # Extract meta data
            meta_data = await self._extract_meta_data(**inputs)
            
            if not meta_data:
                return await self._handle_no_data_fallback()
            
            # Update system performance
            performance_result = await self._update_system_performance(meta_data)
            
            # Evaluate automation decision
            decision_result = await self._evaluate_automation_decision(meta_data)
            
            # Execute decision if needed
            execution_result = {}
            if decision_result.get('decision_required', False):
                execution_result = await self._execute_decision(decision_result, meta_data)
            
            # Update automation metrics
            metrics_result = await self._update_automation_metrics()
            
            # Combine results
            result = {**performance_result, **decision_result, **execution_result, **metrics_result}
            
            # Generate thesis
            thesis = await self._generate_meta_thesis(meta_data, result)
            
            # Update SmartInfoBus
            await self._update_meta_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_meta_error(e, start_time)

    async def _extract_meta_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract meta data from SmartInfoBus"""
        try:
            # Get system performance
            system_performance = self.smart_bus.get('system_performance', 'MetaAgent') or {}
            
            # Get training metrics
            training_metrics = self.smart_bus.get('training_metrics', 'MetaAgent') or {}
            
            # Get market conditions
            market_conditions = self.smart_bus.get('market_conditions', 'MetaAgent') or {}
            
            # Get risk signals
            risk_signals = self.smart_bus.get('risk_signals', 'MetaAgent') or {}
            
            # Get direct inputs
            pnl = inputs.get('pnl', 0.0)
            performance_update = inputs.get('performance_update')
            
            return {
                'system_performance': system_performance,
                'training_metrics': training_metrics,
                'market_conditions': market_conditions,
                'risk_signals': risk_signals,
                'pnl': pnl,
                'performance_update': performance_update,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract meta data: {e}")
            return None

    async def _update_system_performance(self, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update system performance metrics"""
        try:
            # Process PnL update
            pnl = meta_data.get('pnl', 0.0)
            if not np.isnan(pnl) and pnl != 0.0:
                self._update_pnl_metrics(pnl)
            
            # Process performance update
            performance_update = meta_data.get('performance_update')
            if performance_update:
                self._update_performance_from_data(performance_update)
            
            # Process training metrics
            training_metrics = meta_data.get('training_metrics', {})
            if training_metrics:
                self._update_training_metrics(training_metrics)
            
            # Update confidence based on recent performance
            self._update_system_confidence(meta_data)
            
            # Store performance history
            self._performance_history.append({
                'timestamp': time.time(),
                'pnl': self.daily_pnl,
                'confidence': self.system_confidence,
                'mode': self.current_mode.value,
                'drawdown': self.drawdown_pct
            })
            
            return {
                'performance_updated': True,
                'daily_pnl': self.daily_pnl,
                'system_confidence': self.system_confidence,
                'current_mode': self.current_mode.value,
                'drawdown_pct': self.drawdown_pct
            }
            
        except Exception as e:
            self.logger.error(f"Performance update failed: {e}")
            return {'performance_updated': False, 'error': str(e)}

    def _update_pnl_metrics(self, pnl: float):
        """Update PnL-based metrics"""
        self.daily_pnl += pnl
        self.session_pnl += pnl
        
        # Update peak and drawdown
        if self.daily_pnl > self.peak_pnl:
            self.peak_pnl = self.daily_pnl
        
        if self.peak_pnl > 0:
            self.drawdown_pct = (self.peak_pnl - self.daily_pnl) / self.peak_pnl * 100
        else:
            self.drawdown_pct = 0.0
        
        # Update streaks
        if pnl > 0:
            self.win_streak += 1
            self.loss_streak = 0
            self.consecutive_losses = 0
        elif pnl < 0:
            self.loss_streak += 1
            self.win_streak = 0
            self.consecutive_losses += 1

    def _update_performance_from_data(self, performance_data: Dict[str, Any]):
        """Update performance from external data"""
        if 'episode_reward' in performance_data:
            reward = performance_data['episode_reward']
            if reward > self.best_training_reward:
                self.best_training_reward = reward
                self.training_convergence_count += 1
        
        if 'validation_score' in performance_data:
            self.validation_performance.append(performance_data['validation_score'])

    def _update_training_metrics(self, training_metrics: Dict[str, Any]):
        """Update training-related metrics"""
        if 'episodes_completed' in training_metrics:
            self.training_episodes = training_metrics['episodes_completed']
        
        if 'convergence_achieved' in training_metrics:
            if training_metrics['convergence_achieved']:
                self.training_convergence_count += 1

    def _update_system_confidence(self, meta_data: Dict[str, Any]):
        """Update system confidence based on performance"""
        try:
            # Base confidence on recent performance
            if len(self._performance_history) > 0:
                recent_performance = [p['pnl'] for p in list(self._performance_history)[-10:]]
                if recent_performance:
                    avg_performance = np.mean(recent_performance)
                    performance_confidence = max(0, min(1, (avg_performance + 50) / 100))
                else:
                    performance_confidence = 0.5
            else:
                performance_confidence = 0.5
            
            # Factor in drawdown
            drawdown_confidence = max(0, 1 - (self.drawdown_pct / 50))  # 50% drawdown = 0 confidence
            
            # Factor in consecutive losses
            loss_confidence = max(0, 1 - (self.consecutive_losses / 10))  # 10 losses = 0 confidence
            
            # Factor in training performance
            training_confidence = 0.5
            if self.training_episodes > 0:
                training_confidence = min(1, self.training_convergence_count / 10)
            
            # Combine confidences
            new_confidence = (
                performance_confidence * 0.4 +
                drawdown_confidence * 0.3 +
                loss_confidence * 0.2 +
                training_confidence * 0.1
            )
            
            # Apply smoothing
            self.system_confidence = (
                self.config.performance_smoothing * self.system_confidence +
                (1 - self.config.performance_smoothing) * new_confidence
            )
            
            # Apply decay
            self.system_confidence *= self.config.confidence_decay
            
            # Bound confidence
            self.system_confidence = max(0.0, min(1.0, self.system_confidence))
            
            # Store confidence history
            self._confidence_history.append(self.system_confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence update failed: {e}")

    async def _evaluate_automation_decision(self, meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate if automation decision is needed"""
        try:
            current_time = datetime.datetime.now()
            mode_duration = (current_time - self.mode_start_time).total_seconds()
            
            # Check for emergency conditions
            emergency_decision = self._check_emergency_conditions(meta_data)
            if emergency_decision:
                return emergency_decision
            
            # Check mode-specific decision logic
            decision = self._evaluate_mode_specific_decision(meta_data, mode_duration)
            
            return decision or {'decision_required': False}
            
        except Exception as e:
            self.logger.error(f"Decision evaluation failed: {e}")
            return {'decision_required': False, 'error': str(e)}

    def _check_emergency_conditions(self, meta_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Check for emergency stop conditions"""
        # Severe drawdown
        if self.drawdown_pct > 30:  # 30% drawdown
            return {
                'decision_required': True,
                'new_mode': MetaMode.EMERGENCY_STOP,
                'reason': f'Severe drawdown: {self.drawdown_pct:.1f}%',
                'priority': 'critical'
            }
        
        # Consecutive losses
        if self.consecutive_losses >= 15:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EMERGENCY_STOP,
                'reason': f'Excessive consecutive losses: {self.consecutive_losses}',
                'priority': 'high'
            }
        
        # Daily loss threshold
        if self.daily_pnl < self.config.emergency_threshold:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EMERGENCY_STOP,
                'reason': f'Daily loss threshold breached: €{self.daily_pnl:.2f}',
                'priority': 'critical'
            }
        
        # System confidence collapse
        if self.system_confidence < 0.1:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EMERGENCY_STOP,
                'reason': f'System confidence collapsed: {self.system_confidence:.3f}',
                'priority': 'high'
            }
        
        return None

    def _evaluate_mode_specific_decision(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate mode-specific transition logic"""
        if self.current_mode == MetaMode.INITIALIZATION:
            return self._evaluate_initialization_transition(meta_data, mode_duration)
        elif self.current_mode == MetaMode.TRAINING:
            return self._evaluate_training_transition(meta_data, mode_duration)
        elif self.current_mode == MetaMode.VALIDATION:
            return self._evaluate_validation_transition(meta_data, mode_duration)
        elif self.current_mode == MetaMode.LIVE_TRADING:
            return self._evaluate_live_trading_transition(meta_data, mode_duration)
        elif self.current_mode == MetaMode.EVALUATION:
            return self._evaluate_evaluation_transition(meta_data, mode_duration)
        elif self.current_mode == MetaMode.EMERGENCY_STOP:
            return self._evaluate_emergency_transition(meta_data, mode_duration)
        
        return None

    def _evaluate_initialization_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from initialization"""
        if mode_duration > 60:  # 1 minute
            return {
                'decision_required': True,
                'new_mode': MetaMode.TRAINING,
                'reason': 'Initialization complete',
                'priority': 'normal'
            }
        return None

    def _evaluate_training_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from training"""
        # Check if enough episodes completed
        if self.training_episodes >= self.config.min_training_episodes:
            # Check for convergence
            if self.training_convergence_count >= self.config.convergence_episodes:
                return {
                    'decision_required': True,
                    'new_mode': MetaMode.VALIDATION,
                    'reason': f'Training converged after {self.training_episodes} episodes',
                    'priority': 'normal'
                }
        
        # Check for training timeout (prevent infinite training)
        if mode_duration > 7200:  # 2 hours
            return {
                'decision_required': True,
                'new_mode': MetaMode.VALIDATION,
                'reason': 'Training timeout - moving to validation',
                'priority': 'normal'
            }
        
        return None

    def _evaluate_validation_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from validation"""
        if len(self.validation_performance) >= 10:
            avg_validation = np.mean(list(self.validation_performance)[-10:])
            
            if avg_validation > 0.7:  # Good validation performance
                return {
                    'decision_required': True,
                    'new_mode': MetaMode.LIVE_TRADING,
                    'reason': f'Validation successful: {avg_validation:.3f}',
                    'priority': 'normal'
                }
            elif mode_duration > 1800:  # 30 minutes
                return {
                    'decision_required': True,
                    'new_mode': MetaMode.RETRAINING,
                    'reason': f'Validation failed: {avg_validation:.3f}',
                    'priority': 'normal'
                }
        
        return None

    def _evaluate_live_trading_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from live trading"""
        # Check for profit target
        if self.daily_pnl >= self.config.profit_target:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EVALUATION,
                'reason': f'Profit target reached: €{self.daily_pnl:.2f}',
                'priority': 'normal'
            }
        
        # Check for retrain threshold
        if self.daily_pnl <= self.config.retrain_threshold:
            return {
                'decision_required': True,
                'new_mode': MetaMode.RETRAINING,
                'reason': f'Retrain threshold breached: €{self.daily_pnl:.2f}',
                'priority': 'high'
            }
        
        # Check for low confidence
        if self.system_confidence < self.config.confidence_threshold:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EVALUATION,
                'reason': f'Low system confidence: {self.system_confidence:.3f}',
                'priority': 'normal'
            }
        
        return None

    def _evaluate_evaluation_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from evaluation"""
        if mode_duration > self.config.live_evaluation_period:
            if self.system_confidence > self.config.confidence_threshold:
                return {
                    'decision_required': True,
                    'new_mode': MetaMode.LIVE_TRADING,
                    'reason': 'Evaluation complete - confidence restored',
                    'priority': 'normal'
                }
            else:
                return {
                    'decision_required': True,
                    'new_mode': MetaMode.RETRAINING,
                    'reason': 'Evaluation complete - retraining needed',
                    'priority': 'normal'
                }
        
        return None

    def _evaluate_emergency_transition(self, meta_data: Dict[str, Any], mode_duration: float) -> Optional[Dict[str, Any]]:
        """Evaluate transition from emergency stop"""
        if mode_duration > self.config.emergency_cooldown:
            return {
                'decision_required': True,
                'new_mode': MetaMode.EVALUATION,
                'reason': 'Emergency cooldown complete',
                'priority': 'normal'
            }
        
        return None

    async def _execute_decision(self, decision: Dict[str, Any], meta_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute automation decision"""
        try:
            new_mode = decision['new_mode']
            reason = decision['reason']
            priority = decision.get('priority', 'normal')
            
            # Log decision
            self.logger.info(
                format_operator_message(
                    "🔄", "AUTOMATION_DECISION",
                    old_mode=self.current_mode.value,
                    new_mode=new_mode.value,
                    reason=reason,
                    priority=priority,
                    context="automation"
                )
            )
            
            # Record decision
            decision_record = {
                'timestamp': time.time(),
                'old_mode': self.current_mode,
                'new_mode': new_mode,
                'reason': reason,
                'priority': priority,
                'confidence': self.system_confidence,
                'pnl': self.daily_pnl
            }
            self.decision_history.append(decision_record)
            
            # Execute mode transition
            old_mode = self.current_mode
            self._transition_to_mode(new_mode, reason)
            
            # Update metrics
            self.automation_metrics['total_mode_switches'] += 1
            
            if new_mode == MetaMode.EMERGENCY_STOP:
                self.automation_metrics['emergency_stops'] += 1
            elif new_mode == MetaMode.RETRAINING:
                self.automation_metrics['retraining_sessions'] += 1
            elif old_mode == MetaMode.LIVE_TRADING and new_mode != MetaMode.EMERGENCY_STOP:
                self.automation_metrics['successful_live_sessions'] += 1
            
            return {
                'decision_executed': True,
                'old_mode': old_mode.value,
                'new_mode': new_mode.value,
                'reason': reason,
                'priority': priority
            }
            
        except Exception as e:
            self.logger.error(f"Decision execution failed: {e}")
            return {'decision_executed': False, 'error': str(e)}

    def _transition_to_mode(self, new_mode: MetaMode, reason: str):
        """Execute mode transition"""
        old_mode = self.current_mode
        old_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
        
        # Update mode tracking
        self._mode_performance[old_mode.value].append({
            'duration': old_duration,
            'pnl': self.session_pnl,
            'end_confidence': self.system_confidence
        })
        
        # Store transition
        transition = {
            'timestamp': datetime.datetime.now(),
            'from_mode': old_mode,
            'to_mode': new_mode,
            'reason': reason,
            'duration': old_duration
        }
        self.mode_transitions.append(transition)
        
        # Update state
        self.current_mode = new_mode
        self.mode_start_time = datetime.datetime.now()
        self.session_pnl = 0.0  # Reset session PnL
        
        # Mode-specific actions
        if new_mode == MetaMode.TRAINING:
            self.training_start_time = datetime.datetime.now()
            self.training_episodes = 0
        elif new_mode == MetaMode.EVALUATION:
            self.evaluation_start_time = datetime.datetime.now()
        elif new_mode == MetaMode.RETRAINING:
            self.last_retrain_time = datetime.datetime.now()
        elif new_mode == MetaMode.EMERGENCY_STOP:
            self.last_emergency_time = datetime.datetime.now()

    async def _update_automation_metrics(self) -> Dict[str, Any]:
        """Update automation performance metrics"""
        try:
            # Calculate automation score
            if len(self.decision_history) > 0:
                recent_decisions = list(self.decision_history)[-10:]
                decision_quality = self._evaluate_decision_quality(recent_decisions)
                self.automation_score = decision_quality
            else:
                self.automation_score = 0.5
            
            # Update decision confidence
            if len(self._confidence_history) > 0:
                avg_confidence = np.mean(list(self._confidence_history)[-20:])
                self.automation_metrics['decision_confidence'] = avg_confidence
            
            # Calculate automation accuracy
            if self.automation_metrics['total_mode_switches'] > 0:
                success_rate = (
                    self.automation_metrics['successful_live_sessions'] / 
                    max(self.automation_metrics['total_mode_switches'], 1)
                )
                self.automation_metrics['automation_accuracy'] = success_rate
            
            return {
                'automation_metrics': {
                    'automation_score': self.automation_score,
                    'total_mode_switches': self.automation_metrics['total_mode_switches'],
                    'automation_accuracy': self.automation_metrics['automation_accuracy'],
                    'system_confidence': self.system_confidence,
                    'current_mode': self.current_mode.value,
                    'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Automation metrics update failed: {e}")
            return {'automation_metrics': {'error': str(e)}}

    def _evaluate_decision_quality(self, decisions: List[Dict[str, Any]]) -> float:
        """Evaluate quality of recent decisions"""
        if not decisions:
            return 0.5
        
        quality_scores = []
        
        for decision in decisions:
            # Score based on outcome
            if decision['new_mode'] == MetaMode.LIVE_TRADING:
                # Live trading decision - score based on subsequent performance
                score = 0.8 if self.daily_pnl > 0 else 0.3
            elif decision['new_mode'] == MetaMode.EMERGENCY_STOP:
                # Emergency stop - score based on prevention of further losses
                score = 0.9 if decision['priority'] == 'critical' else 0.7
            elif decision['new_mode'] == MetaMode.RETRAINING:
                # Retraining decision - score based on subsequent improvement
                score = 0.6  # Neutral score
            else:
                score = 0.5  # Default score
            
            # Adjust based on timing
            time_since = time.time() - decision['timestamp']
            if time_since < 3600:  # Recent decision
                score *= 1.1  # Slight bonus for recent good decisions
            
            quality_scores.append(score)
        
        return float(np.mean(quality_scores))

    async def _generate_meta_thesis(self, meta_data: Dict[str, Any], 
                                  result: Dict[str, Any]) -> str:
        """Generate comprehensive meta agent thesis"""
        try:
            # Current status
            current_mode = self.current_mode.value
            mode_duration = (datetime.datetime.now() - self.mode_start_time).total_seconds()
            
            # Performance metrics
            daily_pnl = self.daily_pnl
            system_confidence = self.system_confidence
            automation_score = self.automation_score
            
            thesis_parts = [
                f"Meta Agent Status: Operating in {current_mode.upper()} mode for {mode_duration/60:.1f} minutes",
                f"System performance: €{daily_pnl:.2f} daily PnL with {system_confidence:.2f} confidence",
                f"Automation effectiveness: {automation_score:.2f} decision quality score"
            ]
            
            # Decision analysis
            if result.get('decision_executed', False):
                old_mode = result.get('old_mode', 'unknown')
                new_mode = result.get('new_mode', 'unknown')
                reason = result.get('reason', 'unknown')
                thesis_parts.append(f"Mode transition: {old_mode} → {new_mode} ({reason})")
            
            # Risk assessment
            if self.drawdown_pct > 10:
                thesis_parts.append(f"Risk alert: {self.drawdown_pct:.1f}% drawdown detected")
            
            if self.consecutive_losses > 5:
                thesis_parts.append(f"Performance concern: {self.consecutive_losses} consecutive losses")
            
            # Automation metrics
            total_switches = self.automation_metrics['total_mode_switches']
            automation_accuracy = self.automation_metrics['automation_accuracy']
            thesis_parts.append(f"Automation stats: {total_switches} mode switches with {automation_accuracy:.1%} accuracy")
            
            # Training progress
            if self.current_mode in [MetaMode.TRAINING, MetaMode.VALIDATION]:
                thesis_parts.append(f"Training progress: {self.training_episodes} episodes, {self.training_convergence_count} convergences")
            
            # Confidence trend
            if len(self._confidence_history) > 10:
                recent_trend = np.mean(list(self._confidence_history)[-5:]) - np.mean(list(self._confidence_history)[-10:-5])
                trend_desc = "IMPROVING" if recent_trend > 0.01 else "DECLINING" if recent_trend < -0.01 else "STABLE"
                thesis_parts.append(f"Confidence trend: {trend_desc}")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Meta thesis generation failed: {str(e)} - Automation system maintaining core functionality"

    async def _update_meta_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with meta agent results"""
        try:
            # Automation decisions
            automation_data = {
                'current_mode': self.current_mode.value,
                'system_confidence': self.system_confidence,
                'automation_score': self.automation_score,
                'last_decision': result.get('decision_executed', False),
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }
            
            self.smart_bus.set(
                'automation_decisions',
                automation_data,
                module='MetaAgent',
                thesis=thesis
            )
            
            # System mode
            mode_data = {
                'current_mode': self.current_mode.value,
                'mode_start_time': self.mode_start_time.isoformat(),
                'mode_transitions': len(self.mode_transitions),
                'available_modes': [mode.value for mode in MetaMode]
            }
            
            self.smart_bus.set(
                'system_mode',
                mode_data,
                module='MetaAgent',
                thesis="Current system operational mode and transition history"
            )
            
            # Automation metrics
            metrics_data = result.get('automation_metrics', {})
            self.smart_bus.set(
                'automation_metrics',
                metrics_data,
                module='MetaAgent',
                thesis="Meta agent automation performance and decision quality metrics"
            )
            
            # Meta performance
            performance_data = {
                'daily_pnl': self.daily_pnl,
                'drawdown_pct': self.drawdown_pct,
                'consecutive_losses': self.consecutive_losses,
                'win_streak': self.win_streak,
                'system_confidence': self.system_confidence,
                'training_episodes': self.training_episodes
            }
            
            self.smart_bus.set(
                'meta_performance',
                performance_data,
                module='MetaAgent',
                thesis="Meta agent performance tracking and system health metrics"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no meta data is available"""
        self.logger.warning("No meta data available - maintaining current mode")
        
        return {
            'current_mode': self.current_mode.value,
            'system_confidence': self.system_confidence,
            'automation_score': self.automation_score,
            'daily_pnl': self.daily_pnl,
            'fallback_reason': 'no_meta_data'
        }

    async def _handle_meta_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle meta agent errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MetaAgent")
        explanation = self.english_explainer.explain_error(
            "MetaAgent", str(error), "automation decision"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "META_AGENT_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="automation"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'current_mode': self.current_mode.value,
            'system_confidence': self.system_confidence,
            'automation_score': self.automation_score,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_meta_health(self):
        """Update meta agent health metrics"""
        try:
            # Check automation effectiveness
            if self.automation_score < self.config.min_automation_score:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check system confidence
            if self.system_confidence < 0.2:
                self._health_status = 'critical'
            
            # Check emergency conditions
            if self.current_mode == MetaMode.EMERGENCY_STOP:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_automation_effectiveness(self):
        """Analyze automation effectiveness"""
        try:
            if len(self.decision_history) >= 5:
                recent_decisions = list(self.decision_history)[-5:]
                effectiveness = self._evaluate_decision_quality(recent_decisions)
                
                self._decision_effectiveness.append(effectiveness)
                
                if effectiveness > 0.8:
                    self.logger.info(
                        format_operator_message(
                            "🎯", "HIGH_AUTOMATION_EFFECTIVENESS",
                            effectiveness=f"{effectiveness:.2f}",
                            recent_decisions=len(recent_decisions),
                            context="automation_analysis"
                        )
                    )
                elif effectiveness < 0.4:
                    self.logger.warning(
                        format_operator_message(
                            "⚠️", "LOW_AUTOMATION_EFFECTIVENESS",
                            effectiveness=f"{effectiveness:.2f}",
                            recent_decisions=len(recent_decisions),
                            context="automation_analysis"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Automation effectiveness analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MetaAgent', 'automation_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MetaAgent', 'automation_cycle', 0, False
        )

    # Legacy compatibility methods
    def step(self, pnl: float = 0.0, **kwargs):
        """Legacy compatibility for step"""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            meta_data = {'pnl': pnl}
            result = loop.run_until_complete(self.process(**meta_data))
            return result
        finally:
            loop.close()

    def record(self, pnl: float):
        """Legacy compatibility for recording"""
        self._update_pnl_metrics(pnl)

    def force_mode_transition(self, new_mode: str, reason: str = "Manual override") -> bool:
        """Force mode transition"""
        try:
            mode_enum = MetaMode(new_mode)
            self._transition_to_mode(mode_enum, reason)
            
            self.logger.info(
                format_operator_message(
                    "🔧", "FORCED_MODE_TRANSITION",
                    new_mode=new_mode,
                    reason=reason,
                    context="manual_override"
                )
            )
            
            return True
        except ValueError:
            self.logger.error(f"Invalid mode: {new_mode}")
            return False

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'current_mode': self.current_mode.value,
            'automation_metrics': self.automation_metrics.copy(),
            'daily_pnl': self.daily_pnl,
            'system_confidence': self.system_confidence,
            'automation_score': self.automation_score,
            'training_episodes': self.training_episodes,
            'consecutive_losses': self.consecutive_losses,
            'genome': self.genome.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status,
            'mode_start_time': self.mode_start_time.isoformat()
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'current_mode' in state:
            try:
                self.current_mode = MetaMode(state['current_mode'])
            except ValueError:
                self.current_mode = MetaMode.EVALUATION
        
        if 'automation_metrics' in state:
            self.automation_metrics.update(state['automation_metrics'])
        
        if 'daily_pnl' in state:
            self.daily_pnl = state['daily_pnl']
        
        if 'system_confidence' in state:
            self.system_confidence = state['system_confidence']
        
        if 'automation_score' in state:
            self.automation_score = state['automation_score']
        
        if 'training_episodes' in state:
            self.training_episodes = state['training_episodes']
        
        if 'consecutive_losses' in state:
            self.consecutive_losses = state['consecutive_losses']
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']
        
        if 'mode_start_time' in state:
            try:
                self.mode_start_time = datetime.datetime.fromisoformat(state['mode_start_time'])
            except (ValueError, TypeError):
                self.mode_start_time = datetime.datetime.now()

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'system_confidence': self.system_confidence,
            'automation_score': self.automation_score
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        return float(self.system_confidence)

    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        # Meta agent doesn't propose direct actions, but automation decisions
        automation_signal = 1.0 if self.current_mode == MetaMode.LIVE_TRADING else 0.0
        confidence_signal = self.system_confidence
        
        return np.array([automation_signal, confidence_signal])