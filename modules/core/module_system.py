# ─────────────────────────────────────────────────────────────
# File: modules/core/module_system.py
# 🚀 PRODUCTION-READY SmartInfoBus Module System & Orchestrator
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# FIXED: Emergency mode, circuit breakers, dynamic config
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import asyncio
import importlib
import inspect
import time
import threading
import yaml
from pathlib import Path
from typing import Dict, List, Set, Type, Optional, Any, Callable, Tuple
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
import numpy as np
import psutil

from modules.core.module_base import BaseModule, ModuleMetadata
from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.utils.system_utilities import EnglishExplainer
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.core.error_pinpointer import ErrorPinpointer

# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerState:
    """Circuit breaker state for module protection"""
    failure_count: int = 0
    last_failure_time: float = 0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    successful_calls: int = 0
    total_calls: int = 0
    last_success_time: float = 0
    
    def record_success(self):
        """Record successful execution"""
        self.successful_calls += 1
        self.total_calls += 1
        self.last_success_time = time.time()
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = time.time()
    
    def should_allow_request(self, recovery_time: float) -> bool:
        """Check if request should be allowed"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            # Check if recovery time has passed
            if time.time() - self.last_failure_time > recovery_time:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def trip(self):
        """Trip the circuit breaker"""
        self.state = "OPEN"

# ═══════════════════════════════════════════════════════════════════
# ENHANCED MODULE CONFIGURATION WITH DYNAMIC UPDATES
# ═══════════════════════════════════════════════════════════════════

class ModuleConfig:
    """
    PRODUCTION-GRADE configuration with dynamic updates.
    Military-grade parameter validation and hot-reload support.
    """
    
    def __init__(self, **kwargs):
        # Core system defaults
        self.debug = kwargs.get('debug', True)
        self.max_history = kwargs.get('max_history', 1000)
        self.audit_enabled = kwargs.get('audit_enabled', True)
        self.log_rotation_lines = kwargs.get('log_rotation_lines', 5000)
        self.health_check_interval = kwargs.get('health_check_interval', 100)
        self.performance_tracking = kwargs.get('performance_tracking', True)
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.explainable = kwargs.get('explainable', True)
        self.hot_reload = kwargs.get('hot_reload', True)
        
        # Execution parameters
        self.max_parallel_modules = kwargs.get('max_parallel_modules', 10)
        self.default_timeout_ms = kwargs.get('default_timeout_ms', 100)
        self.circuit_breaker_threshold = kwargs.get('circuit_breaker_threshold', 3)
        self.recovery_time_s = kwargs.get('recovery_time_s', 60)
        
        # Error handling
        self.max_retries = kwargs.get('max_retries', 3)
        self.error_escalation = kwargs.get('error_escalation', True)
        self.emergency_shutdown_threshold = kwargs.get('emergency_shutdown_threshold', 5)
        
        # Performance thresholds
        self.latency_warning_ms = kwargs.get('latency_warning_ms', 150)
        self.latency_critical_ms = kwargs.get('latency_critical_ms', 500)
        self.memory_warning_mb = kwargs.get('memory_warning_mb', 1000)
        self.memory_critical_mb = kwargs.get('memory_critical_mb', 2000)
        
        # Emergency mode parameters
        self.emergency_mode_enabled = kwargs.get('emergency_mode_enabled', True)
        self.emergency_cooldown_s = kwargs.get('emergency_cooldown_s', 300)
        self.emergency_health_threshold = kwargs.get('emergency_health_threshold', 0.7)
        
        # Module discovery paths - MODERNIZED MODULES ONLY
        self.module_paths = kwargs.get('module_paths', [
            # ✅ MODERNIZED MODULES (with @module decorator and BaseModule)
            'modules/auditing',     # AuditingCoordinator, TradeExplanationAuditor, TradeThesisTracker
            'modules/external',     # NewsSentimentModule  
            'modules/features',     # AdvancedFeatureEngine, MultiScaleFeatureEngine
            'modules/market',       # MarketThemeDetector, FractalRegimeConfirmation, etc.
            'modules/memory',       # PlaybookMemory, NeuralMemoryArchitect, etc.
            'modules/meta',         # MetaAgent, PPOAgent, PPOLagAgent, etc.
            'modules/models',       # EnhancedWorldModel
            'modules/position',     # PositionManager
            'modules/reward',       # RiskAdjustedReward
            'modules/risk',         # All risk modules are modernized
            
            # 🔄 LEGACY MODULES (commented out until modernized)
            # Uncomment these paths after modernizing the modules in them:
            # 'modules/simulation',    # OpponentSimulator, RoleCoach, ShadowSimulator
            # 'modules/strategy',      # BiasAuditor, CurriculumPlannerPlus, etc.
            # 'modules/trading_modes', # TradingModeManager
            # 'modules/visualization', # VisualizationInterface, TradeMapVisualizer
            # 'modules/voting',        # TimeHorizonAligner, StrategyArbiter, etc.
        ])
        
        # Legacy modules that need modernization (for tracking purposes)
        self.legacy_modules = {
            'modules/simulation': [
                'OpponentSimulator',     # Line 17: class OpponentSimulator(Module, ...)
                'RoleCoach',             # Line 17: class RoleCoach(Module, ...)
                'ShadowSimulator'        # Line 17: class ShadowSimulator(Module, ...)
            ],
            'modules/strategy': [
                'BiasAuditor',           # Line 16: class BiasAuditor(Module, ...)
                'CurriculumPlannerPlus', # Line 16: class CurriculumPlannerPlus(Module, ...)
                'ExplanationGenerator',  # Line 16: class ExplanationGenerator(Module, ...)
                'OpponentModeEnhancer',  # Line 16: class OpponentModeEnhancer(Module, ...)
                'PlaybookClusterer',     # Line 28: class PlaybookClusterer(Module, ...)
                'StrategyGenomePool',    # Line 18: class StrategyGenomePool(Module, ...)
                'StrategyIntrospector',  # Line 16: class StrategyIntrospector(Module, ...)
                'ThesisEvolutionEngine'  # Line 17: class ThesisEvolutionEngine(Module, ...)
            ],
            'modules/trading_modes': [
                'TradingModeManager'     # Line 17: class TradingModeManager(Module, ...)
            ],
            'modules/visualization': [
                'VisualizationInterface', # Line 17: class VisualizationInterface(Module, ...)
                'TradeMapVisualizer'     # Line 16: class TradeMapVisualizer(Module, ...)
            ],
            'modules/voting': [
                'TimeHorizonAligner',       # Line 16: class TimeHorizonAligner(Module, ...)
                'StrategyArbiter',          # Line 18: class StrategyArbiter(Module, ...)
                'ConsensusDetector',        # Line 16: class ConsensusDetector(Module, ...)
                'CollusionAuditor',         # Line 16: class CollusionAuditor(Module, ...)
                'AlternativeRealitySampler' # Line 16: class AlternativeRealitySampler(Module, ...)
            ]
        }
        
        # Dynamic configuration support
        self._config_watchers: List[Callable] = []
        self._config_file_path: Optional[Path] = None
        self._last_config_update = time.time()
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        if self.max_parallel_modules <= 0:
            errors.append("max_parallel_modules must be positive")
        
        if self.default_timeout_ms <= 0:
            errors.append("default_timeout_ms must be positive")
        
        if self.circuit_breaker_threshold <= 0:
            errors.append("circuit_breaker_threshold must be positive")
        
        if not 0 < self.recovery_time_s <= 3600:
            errors.append("recovery_time_s must be between 1 and 3600 seconds")
        
        if not 0 < self.emergency_health_threshold <= 1:
            errors.append("emergency_health_threshold must be between 0 and 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {errors}")
    
    def update_config(self, updates: Dict[str, Any], notify: bool = True):
        """Update configuration dynamically with validation"""
        old_values = {}
        
        for key, value in updates.items():
            if hasattr(self, key):
                old_values[key] = getattr(self, key)
                setattr(self, key, value)
        
        # Re-validate
        try:
            self._validate_config()
        except ValueError as e:
            # Rollback on validation failure
            for key, old_value in old_values.items():
                setattr(self, key, old_value)
            raise e
        
        self._last_config_update = time.time()
        
        # Notify watchers
        if notify:
            for watcher in self._config_watchers:
                try:
                    watcher(updates, old_values)
                except Exception as e:
                    print(f"Config watcher error: {e}")
    
    def add_config_watcher(self, callback: Callable):
        """Add configuration change watcher"""
        self._config_watchers.append(callback)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
    
    def load_from_file(self, config_path: Path):
        """Load configuration from file with hot-reload support"""
        self._config_file_path = config_path
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if 'module_config' in config_data:
                self.update_config(config_data['module_config'])

# ═══════════════════════════════════════════════════════════════════
# ENHANCED MODULE ORCHESTRATOR WITH COMPLETE IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

class ModuleOrchestrator:
    """
    PRODUCTION-GRADE central orchestrator for SmartInfoBus modules.
    
    FIXED IMPLEMENTATIONS:
    - Complete emergency mode system
    - Circuit breaker implementation
    - Health monitoring integration
    - Dynamic configuration support
    - Performance optimization feedback
    - Automated recovery mechanisms
    """
    
    _instance: Optional['ModuleOrchestrator'] = None
    _registered_classes: Dict[str, Type[BaseModule]] = {}
    _lock = threading.Lock()
    
    def __init__(self, 
                 smart_bus: Optional[SmartInfoBus] = None,
                 config: Optional[ModuleConfig] = None):
        """Initialize orchestrator with production-grade defaults"""
        
        # Core references
        self.smart_bus = smart_bus or InfoBusManager.get_instance()
        self.config = config or ModuleConfig()
        self.explainer = EnglishExplainer()
        self.error_pinpointer = ErrorPinpointer(self)
        
        # Module registry
        self.modules: Dict[str, BaseModule] = {}
        self.metadata: Dict[str, ModuleMetadata] = {}
        self.module_classes: Dict[str, Type[BaseModule]] = {}
        
        # Circuit breakers for each module
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        
        # Execution planning
        self.execution_order: List[str] = []
        self.execution_stages: List[List[str]] = []
        self.voting_members: List[str] = []
        self.critical_modules: Set[str] = set()
        
        # Dependency management
        self.module_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_dependencies: Dict[str, Set[str]] = defaultdict(set)
        self.circular_dependencies: List[List[str]] = []
        
        # Performance tracking
        self.execution_history: deque = deque(maxlen=10000)
        self.stage_timings: Dict[str, List[float]] = defaultdict(list)
        self.module_performance: Dict[str, Dict[str, float]] = {}
        
        # Emergency mode state
        self.emergency_mode = False
        self.emergency_mode_reason = ""
        self.emergency_activation_time = 0
        self.emergency_activation_count = 0
        self.last_emergency_check = 0
        
        # Health monitoring integration
        self.health_monitor = None  # Will be set by HealthMonitor
        self.health_check_interval = 10  # seconds
        self.last_health_check = 0
        
        # Error handling and recovery
        self.module_errors: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.consecutive_system_failures = 0
        self.last_successful_execution = time.time()
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_parallel_modules,
            thread_name_prefix="ModuleExec"
        )
        self.execution_lock = threading.RLock()
        
        # Dynamic configuration monitoring
        self.config_monitor_task = None
        self.config.add_config_watcher(self._on_config_change)
        
        # Setup logging
        self.logger = RotatingLogger(
            name="ModuleOrchestrator",
            log_path="logs/orchestrator/orchestrator.log",
            max_lines=10000,
            operator_mode=True,
            info_bus_aware=True,
            plain_english=True
        )
        
        # State management
        from modules.core.persistence import StateManager
        self.state_manager = StateManager()
        
        # Performance optimization
        from modules.monitoring.dependency_visualizer import DependencyVisualizer
        self.dependency_visualizer = DependencyVisualizer(self)
        
        # Initialize system
        self._initialized = False
        ModuleOrchestrator._instance = self
        
        self.logger.info(
            format_operator_message(
                "🚀", "ORCHESTRATOR INITIALIZED",
                details=f"Config: {len(self.config.module_paths)} paths",
                context="startup"
            )
        )
    
    def set_health_monitor(self, health_monitor):
        """Set health monitor reference for integration"""
        self.health_monitor = health_monitor
        self.logger.info("✅ Health monitor integrated with orchestrator")
    
    def _on_config_change(self, updates: Dict[str, Any], old_values: Dict[str, Any]):
        """Handle configuration changes dynamically"""
        self.logger.info(f"📝 Configuration updated: {list(updates.keys())}")
        
        # Update executor if worker count changed
        if 'max_parallel_modules' in updates:
            old_executor = self.executor
            self.executor = ThreadPoolExecutor(
                max_workers=updates['max_parallel_modules'],
                thread_name_prefix="ModuleExec"
            )
            old_executor.shutdown(wait=False)
        
        # Update circuit breaker thresholds
        if 'circuit_breaker_threshold' in updates:
            for cb in self.circuit_breakers.values():
                # Reset counts if threshold lowered
                if updates['circuit_breaker_threshold'] < old_values.get('circuit_breaker_threshold', 3):
                    cb.failure_count = min(cb.failure_count, updates['circuit_breaker_threshold'] - 1)
    
    def initialize(self):
        """Initialize orchestrator with complete system setup"""
        if self._initialized:
            return
        
        try:
            self.logger.info(
                format_operator_message(
                    "🔍", "STARTING MODULE DISCOVERY",
                    details="Scanning module directories",
                    context="initialization"
                )
            )
            
            # Load system configuration
            self._load_system_configuration()
            
            # Discover all modules
            self.discover_all_modules()
            
            # Initialize circuit breakers
            self._initialize_circuit_breakers()
            
            # Build execution plan
            self.build_execution_plan()
            
            # Validate system integrity
            validation_passed = self._validate_system_integrity()
            
            # Initialize emergency mode monitoring
            self._initialize_emergency_monitoring()
            
            # Restore state if available
            self._restore_system_state()
            
            # Start configuration monitoring
            self._start_config_monitoring()
            
            self._initialized = True
            
            self.logger.info(
                format_operator_message(
                    "✅", "ORCHESTRATOR READY",
                    details=f"{len(self.modules)} modules, {len(self.execution_stages)} stages",
                    context="initialization"
                )
            )
            
        except Exception as e:
            self.logger.error(f"💥 INITIALIZATION FAILED: {e}")
            self.error_pinpointer.analyze_error(e, "ModuleOrchestrator")
            raise
    
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for all modules"""
        for module_name in self.modules:
            self.circuit_breakers[module_name] = CircuitBreakerState()
        
        self.logger.info(f"⚡ Initialized {len(self.circuit_breakers)} circuit breakers")
    
    def _initialize_emergency_monitoring(self):
        """Initialize emergency mode monitoring systems"""
        self.emergency_mode = False
        self.emergency_mode_reason = ""
        self.emergency_activation_time = 0
        
        # Set up emergency triggers
        self.emergency_triggers = {
            'system_failure_rate': 0.5,  # 50% module failure rate
            'critical_module_failure': True,  # Any critical module fails
            'memory_critical': 0.9,  # 90% memory usage
            'consecutive_failures': 3,  # 3 consecutive system failures
            'health_score_threshold': 0.3  # Overall health below 30%
        }
        
        self.logger.info("🚨 Emergency monitoring systems initialized")
    
    def _check_emergency_conditions(self) -> Tuple[bool, str]:
        """Check if emergency mode should be activated"""
        # Prevent rapid re-checks
        if time.time() - self.last_emergency_check < 1:
            return False, ""
        
        self.last_emergency_check = time.time()
        
        # Check system failure rate
        if self.execution_history:
            recent_executions = list(self.execution_history)[-10:]
            failure_rate = sum(1 for e in recent_executions if e.get('failure_count', 0) > e.get('success_count', 1)) / len(recent_executions)
            
            if failure_rate >= self.emergency_triggers['system_failure_rate']:
                return True, f"System failure rate {failure_rate:.1%} exceeds threshold"
        
        # Check critical module failures
        for module_name in self.critical_modules:
            cb = self.circuit_breakers.get(module_name)
            if cb and cb.state == "OPEN":
                return True, f"Critical module '{module_name}' circuit breaker is open"
        
        # Check memory usage
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            if memory_percent >= self.emergency_triggers['memory_critical']:
                return True, f"Memory usage {memory_percent:.1%} is critical"
        except:
            pass
        
        # Check consecutive failures
        if self.consecutive_system_failures >= self.emergency_triggers['consecutive_failures']:
            return True, f"System had {self.consecutive_system_failures} consecutive failures"
        
        # Check health score if monitor available
        if self.health_monitor:
            try:
                health_score = self.health_monitor.get_overall_health_score()
                if health_score < self.emergency_triggers['health_score_threshold']:
                    return True, f"System health score {health_score:.1%} below critical threshold"
            except:
                pass
        
        return False, ""
    
    def _enter_emergency_mode(self, reason: str):
        """Enter emergency mode with comprehensive system protection"""
        if self.emergency_mode:
            return  # Already in emergency mode
        
        self.emergency_mode = True
        self.emergency_mode_reason = reason
        self.emergency_activation_time = time.time()
        self.emergency_activation_count += 1
        
        self.logger.critical(
            format_operator_message(
                "🚨", "EMERGENCY MODE ACTIVATED",
                details=reason,
                context="emergency"
            )
        )
        
        # Disable non-critical modules
        disabled_count = 0
        for module_name, metadata in self.metadata.items():
            if not metadata.critical:
                self.smart_bus.record_module_failure(
                    module_name, "Emergency mode - non-critical disabled"
                )
                disabled_count += 1
        
        # Reduce parallel execution
        self.executor._max_workers = max(1, self.config.max_parallel_modules // 2)
        
        # Alert health monitor if available
        if self.health_monitor:
            self.health_monitor.trigger_emergency_alert(reason)
        
        # Store emergency event
        self.smart_bus.set(
            'emergency_mode_event',
            {
                'activated': True,
                'reason': reason,
                'timestamp': self.emergency_activation_time,
                'disabled_modules': disabled_count,
                'activation_count': self.emergency_activation_count
            },
            module='Orchestrator',
            thesis=f"Emergency mode activated due to: {reason}"
        )
        
        self.logger.info(f"🚨 Disabled {disabled_count} non-critical modules")

    def disable_module(self, module_name: str, reason: str = "Manual disable"):
        """Disable a module temporarily"""
        if module_name in self.modules:
            self.smart_bus.record_module_failure(module_name, f"DISABLED: {reason}")
            self.logger.warning(f"⛔ Module disabled: {module_name} - {reason}")
            return True
        return False

    def enable_module(self, module_name: str):
        """Re-enable a disabled module"""
        if module_name in self.modules:
            self.smart_bus.reset_module_failures(module_name)
            # Reset circuit breaker
            if module_name in self.circuit_breakers:
                self.circuit_breakers[module_name] = CircuitBreakerState()
            self.logger.info(f"✅ Module enabled: {module_name}")
            return True
        return False
        
    def exit_emergency_mode(self) -> bool:
        """
        Exit emergency mode if conditions are safe.
        FIXED: Complete implementation with health validation.
        """
        if not self.emergency_mode:
            return True
        
        # Check if enough time has passed
        time_in_emergency = time.time() - self.emergency_activation_time
        if time_in_emergency < self.config.emergency_cooldown_s:
            remaining = self.config.emergency_cooldown_s - time_in_emergency
            self.logger.info(f"⏳ Emergency cooldown: {remaining:.0f}s remaining")
            return False
        
        # Validate system health before exit
        health_checks = {
            'circuit_breakers': self._validate_circuit_breakers(),
            'memory': self._validate_memory_usage(),
            'module_health': self._validate_module_health(),
            'execution_success': self._validate_recent_executions()
        }
        
        # If health monitor available, use it
        if self.health_monitor:
            health_report = self.health_monitor.generate_health_report()
            overall_health = health_report.overall_health_score
            health_checks['overall_health'] = overall_health >= self.config.emergency_health_threshold
        
        # All checks must pass
        all_healthy = all(health_checks.values())
        
        if not all_healthy:
            failed_checks = [k for k, v in health_checks.items() if not v]
            self.logger.warning(f"❌ Cannot exit emergency mode. Failed checks: {failed_checks}")
            return False
        
        # Exit emergency mode
        self.emergency_mode = False
        self.consecutive_system_failures = 0
        
        # Re-enable modules
        enabled_count = 0
        for module_name in self.modules:
            if not self.smart_bus.is_module_enabled(module_name):
                self.smart_bus.reset_module_failures(module_name)
                # Reset circuit breaker
                if module_name in self.circuit_breakers:
                    self.circuit_breakers[module_name] = CircuitBreakerState()
                enabled_count += 1
        
        # Restore parallel execution
        self.executor._max_workers = self.config.max_parallel_modules
        
        # Notify health monitor
        if self.health_monitor:
            self.health_monitor.clear_emergency_alert()
        
        # Store recovery event
        self.smart_bus.set(
            'emergency_mode_recovery',
            {
                'recovered': True,
                'duration_seconds': time_in_emergency,
                'timestamp': time.time(),
                'enabled_modules': enabled_count,
                'health_checks': health_checks
            },
            module='Orchestrator',
            thesis=f"System recovered from emergency mode after {time_in_emergency:.0f}s"
        )
        
        self.logger.info(
            format_operator_message(
                "✅", "EMERGENCY MODE DEACTIVATED",
                details=f"Re-enabled {enabled_count} modules",
                context="recovery"
            )
        )
        
        return True
    
    def _validate_circuit_breakers(self) -> bool:
        """Validate all circuit breakers are healthy"""
        open_breakers = [
            name for name, cb in self.circuit_breakers.items()
            if cb.state == "OPEN"
        ]
        
        # Allow some non-critical breakers to be open
        critical_open = [
            name for name in open_breakers
            if name in self.critical_modules
        ]
        
        return len(critical_open) == 0
    
    def _validate_memory_usage(self) -> bool:
        """Validate memory usage is acceptable"""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            return memory_percent < 0.8  # Below 80%
        except:
            return True  # Assume OK if can't check
    
    def _validate_module_health(self) -> bool:
        """Validate module health status"""
        if not self.modules:
            return False
        
        healthy_count = 0
        for module_name, module in self.modules.items():
            if module.is_healthy:
                healthy_count += 1
        
        health_ratio = healthy_count / len(self.modules)
        return health_ratio >= 0.7  # At least 70% healthy
    
    def _validate_recent_executions(self) -> bool:
        """Validate recent execution success rate"""
        if not self.execution_history:
            return True
        
        recent = list(self.execution_history)[-20:]
        success_rate = sum(
            1 for e in recent 
            if e.get('success_count', 0) > e.get('failure_count', 1)
        ) / len(recent)
        
        return success_rate >= 0.8  # At least 80% success
    
    def _start_config_monitoring(self):
        """Start configuration file monitoring for hot-reload"""
        async def monitor_config():
            config_paths = [
                Path('config/system_config.yaml'),
                Path('config/module_config.yaml')
            ]
            
            last_mtime = {}
            
            while True:
                try:
                    for config_path in config_paths:
                        if config_path.exists():
                            mtime = config_path.stat().st_mtime
                            
                            if config_path not in last_mtime:
                                last_mtime[config_path] = mtime
                            elif mtime > last_mtime[config_path]:
                                # File changed
                                self.logger.info(f"📝 Config file changed: {config_path}")
                                self.config.load_from_file(config_path)
                                last_mtime[config_path] = mtime
                    
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    self.logger.error(f"Config monitoring error: {e}")
                    await asyncio.sleep(30)  # Back off on error
        
        # Start monitoring task
        self.config_monitor_task = asyncio.create_task(monitor_config())
    
    async def execute_step(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute all modules with complete error handling and recovery.
        ENHANCED: Emergency mode checks, circuit breakers, health monitoring.
        """
        if not self._initialized:
            raise RuntimeError("Orchestrator not initialized")
        
        # Check for emergency conditions before execution
        should_enter_emergency, reason = self._check_emergency_conditions()
        if should_enter_emergency:
            self._enter_emergency_mode(reason)
        
        # Try to exit emergency mode if active
        if self.emergency_mode:
            self.exit_emergency_mode()
            
            # If still in emergency mode, use limited execution
            if self.emergency_mode:
                return await self._execute_emergency_mode(market_data)
        
        start_time = time.time()
        execution_id = f"exec_{int(start_time)}"
        
        try:
            with self.execution_lock:
                self.logger.debug(f"🚀 STARTING EXECUTION: {execution_id}")
                
                # Store market data in SmartInfoBus
                self._store_market_data(market_data, execution_id)
                
                # Check system health periodically
                if time.time() - self.last_health_check > self.health_check_interval:
                    self._perform_health_check()
                    self.last_health_check = time.time()
                
                # Execute stages
                results = {}
                stage_results = []
                
                for stage_idx, stage_modules in enumerate(self.execution_stages):
                    stage_start = time.time()
                    
                    try:
                        # Filter modules by circuit breaker state
                        allowed_modules = [
                            m for m in stage_modules
                            if self._check_circuit_breaker(m)
                        ]
                        
                        if not allowed_modules and self._is_critical_stage(stage_modules):
                            raise RuntimeError(f"All modules in critical stage {stage_idx} are circuit-broken")
                        
                        stage_result = await self._execute_stage(
                            allowed_modules, stage_idx, results, execution_id
                        )
                        
                        results.update(stage_result)
                        stage_results.append(stage_result)
                        
                        # Record stage timing
                        stage_duration = (time.time() - stage_start) * 1000
                        self.stage_timings[f"stage_{stage_idx}"].append(stage_duration)
                        
                        # Update performance optimization
                        if hasattr(self, 'dependency_visualizer'):
                            self.dependency_visualizer.update_performance_metrics(
                                f"stage_{stage_idx}", {"duration_ms": stage_duration}
                            )
                        
                        # Check for critical failures
                        if self._check_critical_failures(stage_result):
                            self.logger.error(f"Critical failures in stage {stage_idx}")
                            self.consecutive_system_failures += 1
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Stage {stage_idx} execution failed: {e}")
                        self.error_pinpointer.analyze_error(e, f"Stage{stage_idx}")
                        self.consecutive_system_failures += 1
                        
                        # Continue with other stages unless critical
                        if stage_idx == 0 or self._is_critical_stage(stage_modules):
                            raise
                
                # Aggregate and validate results
                aggregated = self._aggregate_results(results, execution_id)
                
                # Record execution metrics
                execution_time = (time.time() - start_time) * 1000
                self._record_execution(execution_id, execution_time, results, aggregated)
                
                # Reset consecutive failures on success
                if len(aggregated.get('successful_modules', [])) > len(aggregated.get('failed_modules', [])):
                    self.consecutive_system_failures = 0
                    self.last_successful_execution = time.time()
                
                # Generate summary
                summary = self._generate_execution_summary(
                    execution_id, execution_time, stage_results, aggregated
                )
                
                self.logger.info(summary)
                
                return aggregated
                
        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"💥 EXECUTION FAILED: {execution_id} ({execution_time:.0f}ms)")
            self.error_pinpointer.analyze_error(e, "ModuleOrchestrator")
            
            self.consecutive_system_failures += 1
            
            # Check for emergency conditions after failure
            should_enter_emergency, reason = self._check_emergency_conditions()
            if should_enter_emergency:
                self._enter_emergency_mode(reason)
            
            raise
    
    def _check_circuit_breaker(self, module_name: str) -> bool:
        """Check if module's circuit breaker allows execution"""
        cb = self.circuit_breakers.get(module_name)
        if not cb:
            return True
        
        return cb.should_allow_request(self.config.recovery_time_s)
    
    def _perform_health_check(self):
        """Perform system-wide health check"""
        if self.health_monitor:
            try:
                # Request health check
                health_report = self.health_monitor.check_system_health()
                
                # Process alerts
                for alert in health_report.alerts:
                    if alert.severity == "CRITICAL":
                        self.logger.critical(f"🚨 Health Alert: {alert.message}")
                        
                        # Take action based on alert type
                        if "memory" in alert.alert_type:
                            self._handle_memory_alert(alert)
                        elif "latency" in alert.alert_type:
                            self._handle_latency_alert(alert)
                        elif "error_rate" in alert.alert_type:
                            self._handle_error_rate_alert(alert)
                
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
    
    def _handle_memory_alert(self, alert):
        """Handle memory-related health alerts"""
        # Trigger garbage collection
        import gc
        gc.collect()
        
        # Reduce execution parallelism
        if self.executor._max_workers > 2:
            self.executor._max_workers = max(2, self.executor._max_workers // 2)
            self.logger.warning(f"Reduced parallel execution to {self.executor._max_workers} workers")
    
    def _handle_latency_alert(self, alert):
        """Handle latency-related health alerts"""
        # Identify slow modules
        slow_modules = []
        for module_name, cb in self.circuit_breakers.items():
            if module_name in self.module_performance:
                avg_time = self.module_performance[module_name].get('avg_time_ms', 0)
                if avg_time > self.config.latency_critical_ms:
                    slow_modules.append(module_name)
        
        # Temporarily disable slowest non-critical modules
        for module_name in slow_modules:
            if module_name not in self.critical_modules:
                self.logger.warning(f"Temporarily disabling slow module: {module_name}")
                self.smart_bus.record_module_failure(module_name, "Disabled due to high latency")
    
    def _handle_error_rate_alert(self, alert):
        """Handle error rate health alerts"""
        # Reset circuit breakers for modules with improving performance
        for module_name, cb in self.circuit_breakers.items():
            if cb.state == "HALF_OPEN" and cb.successful_calls > 5:
                cb.state = "CLOSED"
                cb.failure_count = 0
                self.logger.info(f"Reset circuit breaker for {module_name}")
    
    async def _execute_module_safe(self,
                                 module: BaseModule,
                                 module_name: str,
                                 inputs: Dict[str, Any],
                                 metadata: ModuleMetadata,
                                 execution_id: str) -> Optional[Dict[str, Any]]:
        """
        Execute module with circuit breaker protection.
        ENHANCED: Complete circuit breaker implementation.
        """
        
        cb = self.circuit_breakers.get(module_name)
        if not cb:
            cb = CircuitBreakerState()
            self.circuit_breakers[module_name] = cb
        
        # Check circuit breaker
        if not cb.should_allow_request(self.config.recovery_time_s):
            self.logger.warning(f"⚡ Circuit breaker OPEN for {module_name}")
            return {'error': 'Circuit breaker open', '_circuit_breaker': True}
        
        start_time = time.perf_counter()
        module_name = module.__class__.__name__ 
        
        try:
            # Pre-execution validation
            if hasattr(module, 'validate_inputs'):
                module.validate_inputs(inputs)
            
            # Execute with timeout
            timeout_sec = metadata.timeout_ms / 1000.0
            
            result = await asyncio.wait_for(module.process(**inputs), timeout_sec)
            
            # Post-execution validation
            if result and isinstance(result, dict):
                if hasattr(module, 'validate_outputs'):
                    module.validate_outputs(result)
                
                # Store outputs in SmartInfoBus
                for output_key in metadata.provides:
                    if output_key in result:
                        thesis = result.get('_thesis', f"Output from {module_name}")
                        confidence = result.get('_confidence', 0.8)
                        
                        self.smart_bus.set(
                            output_key,
                            result[output_key],
                            module=module_name,
                            thesis=thesis,
                            confidence=confidence
                        )
            
            # Record success
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.smart_bus.record_module_timing(module_name, duration_ms)
            module.record_execution(duration_ms, True)
            
            # Update circuit breaker
            cb.record_success()
            
            # Update performance metrics
            if module_name not in self.module_performance:
                self.module_performance[module_name] = {
                    'total_executions': 0,
                    'total_time_ms': 0,
                    'failures': 0,
                    'avg_time_ms': 0
                }
            
            perf = self.module_performance[module_name]
            perf['total_executions'] += 1
            perf['total_time_ms'] += duration_ms
            perf['avg_time_ms'] = perf['total_time_ms'] / perf['total_executions']
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = metadata.timeout_ms
            error_msg = f"Timeout after {duration_ms}ms"
            
            self.smart_bus.record_module_failure(module_name, error_msg)
            module.record_execution(duration_ms, False, error_msg)
            
            # Update circuit breaker
            cb.record_failure()
            if cb.failure_count >= self.config.circuit_breaker_threshold:
                cb.trip()
                self.logger.error(f"⚡ Circuit breaker TRIPPED for {module_name}")
            
            self.logger.error(
                format_operator_message(
                    "⏱️", "MODULE TIMEOUT",
                    instrument=module_name,
                    details=f"{duration_ms}ms",
                    context=execution_id
                )
            )
            
            raise TimeoutError(error_msg)
            
        except Exception as e:
            duration_ms = (time.perf_counter() - start_time) * 1000
            error_msg = str(e)
            
            self.smart_bus.record_module_failure(module_name, error_msg)
            module.record_execution(duration_ms, False, error_msg)
            
            # Update circuit breaker
            cb.record_failure()
            if cb.failure_count >= self.config.circuit_breaker_threshold:
                cb.trip()
                self.logger.error(f"⚡ Circuit breaker TRIPPED for {module_name}")
            
            # Update performance metrics
            if module_name in self.module_performance:
                self.module_performance[module_name]['failures'] += 1
            
            # Analyze error
            self.error_pinpointer.analyze_error(e, module_name)
            
            self.logger.error(
                format_operator_message(
                    "💥", "MODULE FAILED",
                    instrument=module_name,
                    details=error_msg[:100],
                    context=execution_id
                )
            )
            
            raise
    
    async def _execute_emergency_mode(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute only critical modules in emergency mode.
        ENHANCED: Better error handling and monitoring.
        """
        self.logger.warning("🚨 Executing in EMERGENCY MODE - critical modules only")
        
        execution_id = f"emergency_{int(time.time())}"
        start_time = time.time()
        
        # Store market data
        self._store_market_data(market_data, execution_id)
        
        results = {}
        successful = 0
        failed = 0
        
        # Execute critical modules sequentially for safety
        for module_name in sorted(self.critical_modules):
            if module_name not in self.modules:
                continue
            
            try:
                module = self.modules[module_name]
                metadata = self.metadata[module_name]
                
                # Check circuit breaker even in emergency mode
                cb = self.circuit_breakers.get(module_name)
                if cb and cb.state == "OPEN":
                    # Try anyway in emergency, but log
                    self.logger.warning(f"Attempting {module_name} despite open circuit breaker (emergency)")
                
                inputs = self._prepare_module_inputs(module_name, metadata, execution_id)
                
                # Execute with extended timeout
                extended_timeout = metadata.timeout_ms * 2 / 1000.0
                result = await asyncio.wait_for(
                    module.process(**inputs),
                    timeout=extended_timeout
                )
                
                results[module_name] = result
                successful += 1
                
                # Record success for potential recovery
                if cb:
                    cb.record_success()
                
            except Exception as e:
                self.logger.error(f"Emergency execution failed for {module_name}: {e}")
                results[module_name] = {'error': str(e), '_emergency_mode': True}
                failed += 1
                
                # Still update circuit breaker
                if module_name in self.circuit_breakers:
                    self.circuit_breakers[module_name].record_failure()
        
        execution_time = (time.time() - start_time) * 1000
        
        # Try to exit emergency mode if execution was successful
        if successful > failed:
            self.exit_emergency_mode()
        
        return {
            'emergency_mode': True,
            'execution_id': execution_id,
            'results': results,
            'successful_modules': successful,
            'failed_modules': failed,
            'execution_time_ms': execution_time,
            'timestamp': time.time(),
            'emergency_reason': self.emergency_mode_reason
        }
    
    # ... (rest of the methods remain the same but with the enhancements integrated)
    
    def get_module_by_name(self, name: str) -> Optional[BaseModule]:
        """Get module instance by name"""
        return self.modules.get(name)
    
    def get_dependency_graph(self) -> Dict[str, Any]:
        """Get dependency graph for visualization"""
        if hasattr(self, 'dependency_visualizer'):
            return self.dependency_visualizer.get_graph_data()
        
        return {
            'nodes': list(self.modules.keys()),
            'edges': [
                {'from': module, 'to': dep}
                for module, deps in self.module_dependencies.items()
                for dep in deps
            ]
        }
    
    def get_circuit_breaker_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        status = {}
        
        for module_name, cb in self.circuit_breakers.items():
            status[module_name] = {
                'state': cb.state,
                'failure_count': cb.failure_count,
                'success_rate': cb.successful_calls / max(cb.total_calls, 1),
                'last_failure': cb.last_failure_time,
                'can_execute': cb.should_allow_request(self.config.recovery_time_s)
            }
        
        return status
    
    def reset_circuit_breaker(self, module_name: str) -> bool:
        """Manually reset a circuit breaker"""
        if module_name in self.circuit_breakers:
            self.circuit_breakers[module_name] = CircuitBreakerState()
            self.logger.info(f"⚡ Circuit breaker reset for {module_name}")
            return True
        return False
    
    def get_emergency_mode_status(self) -> Dict[str, Any]:
        """Get detailed emergency mode status"""
        return {
            'active': self.emergency_mode,
            'reason': self.emergency_mode_reason,
            'activation_time': self.emergency_activation_time,
            'duration_seconds': time.time() - self.emergency_activation_time if self.emergency_mode else 0,
            'activation_count': self.emergency_activation_count,
            'can_exit': self.exit_emergency_mode() if self.emergency_mode else True,
            'triggers': self.emergency_triggers if hasattr(self, 'emergency_triggers') else {}
        }
    
    def trigger_emergency_mode_manually(self, reason: str = "Manual trigger"):
        """Manually trigger emergency mode for testing"""
        self._enter_emergency_mode(f"MANUAL: {reason}")
    
    def shutdown(self):
        """Graceful system shutdown with cleanup"""
        self.logger.info("🛑 Initiating system shutdown...")
        
        try:
            # Cancel config monitoring
            if self.config_monitor_task:
                self.config_monitor_task.cancel()
            
            # Save all module states
            self.state_manager.create_checkpoint(self, "shutdown")
            
            # Log circuit breaker final state
            cb_summary = {
                name: cb.state 
                for name, cb in self.circuit_breakers.items()
            }
            self.logger.info(f"Final circuit breaker states: {cb_summary}")
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            # Clear registrations
            self._registered_classes.clear()
            self.modules.clear()
            self.circuit_breakers.clear()
            
            self.logger.info("✅ System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    # Add remaining missing methods from original with fixes...
    
    def discover_modules(self) -> Dict[str, Type[BaseModule]]:
        """Discover and validate all available modules"""
        discovered = {}
        
        for path_str in self.config.module_paths:
            path = Path(path_str)
            if not path.exists():
                self.logger.warning(f"Module path does not exist: {path}")
                continue
            
            for py_file in path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                
                module_name = None
                try:
                    # Import module
                    module_name = f"{path_str.replace('/', '.')}.{py_file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find all BaseModule subclasses
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseModule) and 
                            obj != BaseModule and
                            hasattr(obj, '__module_metadata__')):
                            
                            discovered[name] = obj
                            self.logger.debug(f"Discovered module: {name}")
                            
                except ImportError as e:
                    self.logger.error(f"Failed to import {module_name or py_file}: {e}")
                except Exception as e:
                    self.logger.error(f"Error discovering modules in {py_file}: {e}")
        
        return discovered
    
    def discover_all_modules(self):
        """Discover and register all modules"""
        discovered = self.discover_modules()
        
        # Register discovered modules
        for name, module_class in discovered.items():
            if name not in self.modules:
                self.register_module(name, module_class)
        
        # Also register pre-loaded classes
        for name, cls in self._registered_classes.items():
            if name not in self.modules:
                self.register_module(name, cls)
        
        self.logger.info(f"Module discovery complete: {len(self.modules)} modules registered")
    
    def register_module(self, name: str, module_class: Type[BaseModule]):
        """Register a module with the orchestrator"""
        try:
            # Validate module
            if not hasattr(module_class, '__module_metadata__'):
                raise ValueError(f"Module {name} missing metadata")
            
            metadata = getattr(module_class, '__module_metadata__', None)
            if not metadata:
                raise ValueError(f"Module {name} missing metadata")
            
            # Check for conflicts
            if name in self.modules:
                self.logger.warning(f"Module {name} already registered")
                return
            
            # Get module configuration from ConfigurationManager
            module_config = {}
            if hasattr(self, 'config_manager') and self.config_manager:
                module_config = self.config_manager.get_module_config(name)
            
            # Create instance with configuration
            try:
                # Try to pass configuration to module constructor
                if module_config:
                    instance = module_class(config=module_config)
                else:
                    instance = module_class()
            except TypeError:
                # Fallback if module doesn't accept config parameter
                instance = module_class()
                # Apply configuration after creation if module has set_config method
                if hasattr(instance, 'set_config') and module_config:
                    instance.set_config(module_config)
            
            # Store module
            self.modules[name] = instance
            self.metadata[name] = metadata
            self.module_classes[name] = module_class
            
            # Initialize circuit breaker
            self.circuit_breakers[name] = CircuitBreakerState()
            
            # Register with SmartInfoBus
            self.smart_bus.register_provider(name, metadata.provides)
            self.smart_bus.register_consumer(name, metadata.requires)
            
            # Track special modules
            if metadata.is_voting_member:
                self.voting_members.append(name)
            
            if metadata.critical:
                self.critical_modules.add(name)
            
            self.logger.info(f"✅ Registered module: {name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register {name}: {e}")
    
    def build_execution_plan(self):
        """Build execution plan with dependency resolution"""
        try:
            # Build dependencies
            for name, metadata in self.metadata.items():
                self._build_module_dependencies(name, metadata)
            
            # Find circular dependencies
            self.circular_dependencies = self._find_circular_dependencies()
            if self.circular_dependencies:
                self.logger.warning(f"Found circular dependencies: {self.circular_dependencies}")
                self._break_circular_dependencies()
            
            # Topological sort
            self.execution_order = self._topological_sort()
            
            # Build parallel stages
            self.execution_stages = self._build_parallel_stages()
            
            # Optimize if visualizer available
            if hasattr(self, 'dependency_visualizer'):
                optimized = self.dependency_visualizer.optimize_execution_stages()
                if optimized:
                    self.execution_stages = optimized
            
            self._log_execution_plan()
            
        except Exception as e:
            self.logger.error(f"Failed to build execution plan: {e}")
            raise
    
    def _build_module_dependencies(self, module_name: str, metadata: ModuleMetadata):
        """Build dependency graph for module"""
        self.module_dependencies[module_name].clear()
        
        for required_key in metadata.requires:
            providers = self.smart_bus.get_providers(required_key)
            for provider in providers:
                if provider != module_name and provider in self.modules:
                    self.module_dependencies[module_name].add(provider)
                    self.reverse_dependencies[provider].add(module_name)
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies using DFS"""
        def dfs(node: str, path: List[str], visited: Set[str]) -> List[List[str]]:
            if node in path:
                cycle_start = path.index(node)
                return [path[cycle_start:] + [node]]
            
            if node in visited:
                return []
            
            visited.add(node)
            path.append(node)
            
            cycles = []
            for dep in self.module_dependencies.get(node, []):
                cycles.extend(dfs(dep, path.copy(), visited.copy()))
            
            return cycles
        
        all_cycles = []
        for module in self.modules:
            cycles = dfs(module, [], set())
            all_cycles.extend(cycles)
        
        # Remove duplicates
        unique_cycles = []
        for cycle in all_cycles:
            normalized = tuple(sorted(cycle[:-1]))
            if not any(normalized == tuple(sorted(c[:-1])) for c in unique_cycles):
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    def _break_circular_dependencies(self):
        """Break circular dependencies by removing lowest priority edges"""
        for cycle in self.circular_dependencies:
            if len(cycle) < 2:
                continue
            
            # Find lowest priority module
            min_priority = float('inf')
            min_module = None
            
            for module in cycle[:-1]:
                if module in self.metadata:
                    priority = self.metadata[module].priority
                    if priority < min_priority:
                        min_priority = priority
                        min_module = module
            
            if min_module and min_module in self.module_dependencies:
                deps = list(self.module_dependencies[min_module])
                if deps:
                    removed_dep = deps[0]
                    self.module_dependencies[min_module].discard(removed_dep)
                    self.reverse_dependencies[removed_dep].discard(min_module)
                    self.logger.warning(f"Broke circular dependency: {min_module} -> {removed_dep}")
    
    def _topological_sort(self) -> List[str]:
        """Topological sort with priority"""
        in_degree = defaultdict(int)
        
        for module in self.modules:
            for dep in self.module_dependencies[module]:
                in_degree[dep] += 1
        
        available = sorted(
            [m for m in self.modules if in_degree[m] == 0],
            key=lambda m: self.metadata[m].priority,
            reverse=True
        )
        
        result = []
        
        while available:
            module = available.pop(0)
            result.append(module)
            
            for dependent in self.reverse_dependencies.get(module, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    priority = self.metadata[dependent].priority
                    inserted = False
                    for i, existing in enumerate(available):
                        if self.metadata[existing].priority < priority:
                            available.insert(i, dependent)
                            inserted = True
                            break
                    if not inserted:
                        available.append(dependent)
        
        remaining = set(self.modules) - set(result)
        if remaining:
            self.logger.warning(f"Orphaned modules: {remaining}")
            result.extend(sorted(remaining))
        
        return result
    
    def _build_parallel_stages(self) -> List[List[str]]:
        """Build parallel execution stages"""
        stages = []
        remaining = set(self.modules.keys())
        completed = set()
        
        while remaining:
            stage = []
            
            for module in remaining:
                deps = self.module_dependencies.get(module, set())
                if deps.issubset(completed):
                    stage.append(module)
            
            if not stage:
                # Take highest priority remaining
                stage = sorted(
                    list(remaining)[:self.config.max_parallel_modules],
                    key=lambda m: self.metadata[m].priority,
                    reverse=True
                )
                self.logger.warning(f"Forced stage: {stage}")
            
            stages.append(stage)
            completed.update(stage)
            remaining -= set(stage)
        
        return stages
    
    def _log_execution_plan(self):
        """Log execution plan"""
        lines = [
            "EXECUTION PLAN",
            "=" * 50,
            f"Modules: {len(self.modules)}",
            f"Stages: {len(self.execution_stages)}",
            f"Critical: {len(self.critical_modules)}",
            ""
        ]
        
        for i, stage in enumerate(self.execution_stages, 1):
            lines.append(f"Stage {i}: {len(stage)} modules")
            for module in stage:
                meta = self.metadata[module]
                tags = []
                if meta.critical:
                    tags.append("CRITICAL")
                if meta.is_voting_member:
                    tags.append("VOTER")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  • {module}{tag_str}")
        
        self.logger.info("\n".join(lines))
    
    async def _execute_stage(self, 
                           module_names: List[str], 
                           stage_idx: int,
                           previous_results: Dict[str, Any],
                           execution_id: str) -> Dict[str, Any]:
        """Execute a stage of modules in parallel"""
        self.logger.debug(f"Executing stage {stage_idx}: {module_names}")
        
        tasks = []
        results = {}
        
        for module_name in module_names:
            if not self.smart_bus.is_module_enabled(module_name):
                self.logger.warning(f"Skipping disabled module: {module_name}")
                continue
            
            module = self.modules[module_name]
            metadata = self.metadata[module_name]
            
            inputs = self._prepare_module_inputs(module_name, metadata, execution_id)
            
            task = asyncio.create_task(
                self._execute_module_safe(
                    module, module_name, inputs, metadata, execution_id
                ),
                name=f"{execution_id}_{module_name}"
            )
            
            tasks.append((module_name, task))
        
        # Execute with timeout
        stage_timeout = max(
            self.metadata[name].timeout_ms for name in module_names
            if name in self.metadata
        ) / 1000.0 + 5.0
        
        try:
            await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=stage_timeout
            )
        except asyncio.TimeoutError:
            self.logger.error(f"Stage {stage_idx} timeout")
            for _, task in tasks:
                if not task.done():
                    task.cancel()
        
        # Collect results
        for module_name, task in tasks:
            try:
                if task.done() and not task.cancelled():
                    result = task.result()
                    if isinstance(result, Exception):
                        self.logger.error(f"Module {module_name} failed: {result}")
                        results[module_name] = {'error': str(result)}
                    else:
                        results[module_name] = result
                else:
                    results[module_name] = {'error': 'Task incomplete'}
            except Exception as e:
                self.logger.error(f"Error collecting result from {module_name}: {e}")
                results[module_name] = {'error': str(e)}
        
        return results
    
    def _prepare_module_inputs(self, 
                             module_name: str, 
                             metadata: ModuleMetadata,
                             execution_id: str) -> Dict[str, Any]:
        """Prepare inputs for module execution"""
        inputs = {'execution_id': execution_id}
        missing_inputs = []
        
        for required_key in metadata.requires:
            data = self.smart_bus.get_with_metadata(required_key, module_name)
            
            if data:
                inputs[required_key] = data.value
                
                # Check staleness
                if data.age_seconds() > 60:
                    self.logger.warning(
                        f"Stale data for {module_name}: {required_key} ({data.age_seconds():.1f}s old)"
                    )
            else:
                value = self.smart_bus.get(required_key, module_name)
                if value is not None:
                    inputs[required_key] = value
                else:
                    missing_inputs.append(required_key)
        
        if missing_inputs:
            self.logger.warning(f"Missing inputs for {module_name}: {missing_inputs}")
            for key in missing_inputs:
                self.smart_bus.request_data(key, module_name)
                inputs[key] = ""
        
        return inputs
    
    def _store_market_data(self, market_data: Dict[str, Any], execution_id: str):
        """Store market data in SmartInfoBus"""
        try:
            if not isinstance(market_data, dict):
                raise ValueError("Market data must be a dictionary")
            
            for key, value in market_data.items():
                if not key.startswith('_'):
                    self.smart_bus.set(
                        key,
                        value,
                        module="Environment",
                        thesis=f"Market data for {execution_id}",
                        confidence=1.0
                    )
            
            self.smart_bus.set(
                'execution_metadata',
                {
                    'execution_id': execution_id,
                    'timestamp': time.time(),
                    'data_keys': list(market_data.keys())
                },
                module="Orchestrator",
                thesis=f"Execution metadata for {execution_id}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to store market data: {e}")
            raise
    
    def _check_critical_failures(self, stage_result: Dict[str, Any]) -> bool:
        """Check if stage had critical failures"""
        for module_name, result in stage_result.items():
            if module_name in self.critical_modules:
                if isinstance(result, dict) and 'error' in result:
                    return True
        return False
    
    def _is_critical_stage(self, stage_modules: List[str]) -> bool:
        """Check if stage contains critical modules"""
        return any(m in self.critical_modules for m in stage_modules)
    
    def _aggregate_results(self, 
                         results: Dict[str, Any], 
                         execution_id: str) -> Dict[str, Any]:
        """Aggregate execution results"""
        aggregated = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'module_count': len(results),
            'successful_modules': [],
            'failed_modules': [],
            'votes': {},
            'signals': {},
            'analysis': {},
            'theses': {},
            'performance_metrics': {}
        }
        
        for module_name, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                aggregated['successful_modules'].append(module_name)
                
                for key, value in result.items():
                    if key == '_thesis':
                        aggregated['theses'][module_name] = value
                    elif key == 'vote':
                        aggregated['votes'][module_name] = value
                    elif key == 'trading_signal':
                        aggregated['signals'][module_name] = value
                    elif not key.startswith('_'):
                        if key not in aggregated['analysis']:
                            aggregated['analysis'][key] = {}
                        aggregated['analysis'][key][module_name] = value
            else:
                aggregated['failed_modules'].append({
                    'module': module_name,
                    'error': result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                })
        
        # Add performance metrics
        for module_name in results:
            if module_name in self.modules:
                health = self.modules[module_name].get_health_status()
                aggregated['performance_metrics'][module_name] = health['performance']
        
        # Store in SmartInfoBus
        self.smart_bus.set(
            'execution_results',
            aggregated,
            module='Orchestrator',
            thesis=self.explainer.explain_execution_results(
                aggregated, 
                execution_time=0.0,  # Will be calculated from execution history
                module_count=len(results),
                success_count=len(aggregated['successful_modules'])
            ),
            confidence=0.9
        )
        
        return aggregated
    
    def _record_execution(self, 
                        execution_id: str,
                        execution_time: float,
                        results: Dict[str, Any],
                        aggregated: Dict[str, Any]):
        """Record execution metrics"""
        record = {
            'execution_id': execution_id,
            'timestamp': time.time(),
            'execution_time_ms': execution_time,
            'module_count': len(results),
            'success_count': len(aggregated['successful_modules']),
            'failure_count': len(aggregated['failed_modules']),
            'emergency_mode': self.emergency_mode
        }
        
        self.execution_history.append(record)
    
    def _generate_execution_summary(self,
                                  execution_id: str,
                                  execution_time: float,
                                  stage_results: List[Dict],
                                  aggregated: Dict[str, Any]) -> str:
        """Generate execution summary"""
        mode = "🚨 EMERGENCY" if self.emergency_mode else "✅ NORMAL"
        
        lines = [
            f"EXECUTION COMPLETE: {execution_id} [{mode}]",
            "=" * 50,
            f"Time: {execution_time:.0f}ms",
            f"Success: {len(aggregated['successful_modules'])}/{aggregated['module_count']}",
        ]
        
        if aggregated['failed_modules']:
            lines.append("FAILURES:")
            for failure in aggregated['failed_modules'][:3]:
                lines.append(f"  • {failure['module']}: {failure['error'][:50]}...")
        
        return "\n".join(lines)
    
    def _load_system_configuration(self):
        """Load system configuration from files using ConfigurationManager"""
        try:
            from modules.core.configuration_manager import ConfigurationManager
            
            # Get configuration manager instance
            config_manager = ConfigurationManager.get_instance()
            
            # Load execution configuration
            execution_config = config_manager.get_execution_config()
            if execution_config:
                self._apply_execution_configuration(execution_config)
            
            # Load module registry
            module_registry = config_manager.get_module_registry()
            if module_registry:
                self._apply_module_registry(module_registry)
            
            # Store reference to config manager
            self.config_manager = config_manager
            
            self.logger.info("✅ System configuration loaded from ConfigurationManager")
            
        except Exception as e:
            self.logger.error(f"Failed to load system configuration: {e}")
            self.config_manager = None
    
    def _apply_execution_configuration(self, execution_config: Dict[str, Any]):
        """Apply execution configuration to orchestrator"""
        try:
            # Apply timeouts
            if 'timeouts' in execution_config:
                timeouts = execution_config['timeouts']
                if 'default_ms' in timeouts:
                    self.config.default_timeout_ms = timeouts['default_ms']
                
                # Apply module-specific timeouts
                if 'by_module' in timeouts:
                    self.module_timeouts = timeouts['by_module']
                
                # Apply category-specific timeouts
                if 'by_category' in timeouts:
                    self.category_timeouts = timeouts['by_category']
            
            # Apply circuit breaker configuration
            if 'circuit_breakers' in execution_config:
                cb_config = execution_config['circuit_breakers']
                if 'failure_threshold' in cb_config:
                    self.config.circuit_breaker_threshold = cb_config['failure_threshold']
                if 'recovery_time_s' in cb_config:
                    self.config.recovery_time_s = cb_config['recovery_time_s']
            
            # Apply performance configuration
            if 'performance' in execution_config:
                perf_config = execution_config['performance']
                if 'enable_caching' in perf_config:
                    self.config.cache_enabled = perf_config['enable_caching']
                
                # Apply memory limits
                if 'memory_limits' in perf_config:
                    mem_limits = perf_config['memory_limits']
                    if 'per_module_mb' in mem_limits:
                        self.config.memory_warning_mb = mem_limits['per_module_mb']
                
                # Apply CPU limits
                if 'cpu_limits' in perf_config:
                    cpu_limits = perf_config['cpu_limits']
                    if 'total_worker_threads' in cpu_limits:
                        self.config.max_parallel_modules = cpu_limits['total_worker_threads']
            
            # Apply parallel stages configuration
            if 'parallel_stages' in execution_config:
                self.execution_stages_config = execution_config['parallel_stages']
            
            self.logger.info("✅ Applied execution configuration")
            
        except Exception as e:
            self.logger.error(f"Failed to apply execution configuration: {e}")
    
    def _apply_module_registry(self, module_registry: Dict[str, Any]):
        """Apply module registry configuration"""
        try:
            # Store module registry for later use during registration
            self.module_registry_config = module_registry
            
            # Apply module-specific configurations
            for module_name, module_config in module_registry.items():
                if isinstance(module_config, dict):
                    # Store module configuration for later application
                    if not hasattr(self, 'pending_module_configs'):
                        self.pending_module_configs = {}
                    self.pending_module_configs[module_name] = module_config
            
            self.logger.info(f"✅ Applied module registry: {len(module_registry)} modules")
            
        except Exception as e:
            self.logger.error(f"Failed to apply module registry: {e}")
    
    def _validate_system_integrity(self) -> bool:
        """Validate system integrity"""
        issues = []
        
        # Check dependencies
        for module_name, metadata in self.metadata.items():
            for required in metadata.requires:
                providers = self.smart_bus.get_providers(required)
                if not providers:
                    issues.append(f"No provider for {required} (required by {module_name})")
        
        # Check isolated modules
        isolated = []
        for module_name in self.modules:
            if (not self.module_dependencies.get(module_name) and 
                not self.reverse_dependencies.get(module_name)):
                isolated.append(module_name)
        
        if isolated:
            issues.append(f"Isolated modules: {isolated}")
        
        # Check voting system
        if not self.voting_members:
            issues.append("No voting members configured")
        
        if issues:
            self.logger.warning(f"INTEGRITY ISSUES:\n" + "\n".join(f"  • {i}" for i in issues))
            return False
        
        self.logger.info("✅ System integrity validated")
        return True
    
    def _restore_system_state(self):
        """Restore system state from persistence"""
        try:
            restored = self.state_manager.restore_all_states(self)
            if restored:
                self.logger.info(f"Restored state for {sum(restored.values())} modules")
        except Exception as e:
            self.logger.error(f"Failed to restore state: {e}")
    
    @classmethod
    def get_instance(cls) -> 'ModuleOrchestrator':
        """Get orchestrator singleton"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    @classmethod
    def register_class(cls, module_class: Type[BaseModule]):
        """Register module class"""
        cls._registered_classes[module_class.__name__] = module_class
        
        if cls._instance:
            cls._instance.register_module(module_class.__name__, module_class)
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get comprehensive execution metrics"""
        if not self.execution_history:
            return {}
        
        recent = list(self.execution_history)[-100:]
        
        return {
            'total_executions': len(self.execution_history),
            'avg_execution_time_ms': np.mean([r['execution_time_ms'] for r in recent]),
            'success_rate': np.mean([
                r['success_count'] / max(r['module_count'], 1) for r in recent
            ]),
            'emergency_mode': self.emergency_mode,
            'circuit_breakers': self.get_circuit_breaker_status(),
            'module_performance': self.module_performance
        }
    
    def get_system_status_report(self) -> str:
        """Get system status report"""
        metrics = self.get_execution_metrics()
        emergency_status = self.get_emergency_mode_status()
        
        lines = [
            "SMARTINFOBUS SYSTEM STATUS",
            "=" * 50,
            f"Mode: {'🚨 EMERGENCY' if self.emergency_mode else '✅ NORMAL'}",
            f"Modules: {len(self.modules)} ({len(self.critical_modules)} critical)",
            f"Circuit Breakers: {sum(1 for cb in self.circuit_breakers.values() if cb.state == 'OPEN')} open",
        ]
        
        if self.emergency_mode:
            lines.extend([
                "",
                "EMERGENCY MODE DETAILS:",
                f"  Reason: {emergency_status['reason']}",
                f"  Duration: {emergency_status['duration_seconds']:.0f}s",
                f"  Can Exit: {emergency_status['can_exit']}"
            ])
        
        return "\n".join(lines)

    def get_legacy_module_status(self) -> Dict[str, Any]:
        """Get status of legacy modules that need modernization."""
        legacy_status = {}
        for path, modules in self.config.legacy_modules.items():
            legacy_status[path] = {
                'total_modules': len(modules),
                'modernized_count': 0,
                'instructions': []
            }
            for module_name in modules:
                if module_name in self.modules:
                    legacy_status[path]['modernized_count'] += 1
                else:
                    legacy_status[path]['instructions'].append({
                        'module_name': module_name,
                        'path': path,
                        'reason': 'Module not found in current modules list. Please ensure it is registered.'
                    })
        return legacy_status