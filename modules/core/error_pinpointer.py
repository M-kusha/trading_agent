# ─────────────────────────────────────────────────────────────
# File: modules/core/error_pinpointer.py
# 🚀 PRODUCTION-READY Error Analysis & Debugging System
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# FIXED: Recovery integration, error correlation, action automation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import asyncio
import inspect
import re
import time
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import psutil
import threading

from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE ERROR ANALYSIS STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ErrorContext:
    """
    Comprehensive error context for precise debugging.
    Military-grade error analysis with full system state.
    """
    error_type: str
    error_message: str
    module_name: str
    function_name: str
    file_path: str
    line_number: int
    timestamp: datetime
    
    # Code context
    source_lines: List[str] = field(default_factory=list)
    local_variables: Dict[str, Any] = field(default_factory=dict)
    call_stack: List[Dict[str, Any]] = field(default_factory=list)
    
    # System context
    module_state: Dict[str, Any] = field(default_factory=dict)
    infobus_snapshot: Dict[str, Any] = field(default_factory=dict)
    related_errors: List[str] = field(default_factory=list)
    
    # Analysis
    severity: str = "unknown"  # critical, high, medium, low
    category: str = "unknown"  # logic, data, timeout, dependency, resource
    suggested_fixes: List[str] = field(default_factory=list)
    reproduction_steps: List[str] = field(default_factory=list)
    
    # Recovery actions
    recovery_actions: List[Dict[str, Any]] = field(default_factory=list)
    action_taken: bool = False
    action_result: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'module_name': self.module_name,
            'function_name': self.function_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'timestamp': self.timestamp.isoformat(),
            'source_lines': self.source_lines,
            'local_variables': self.local_variables,
            'call_stack': self.call_stack,
            'module_state': self.module_state,
            'infobus_snapshot': self.infobus_snapshot,
            'related_errors': self.related_errors,
            'severity': self.severity,
            'category': self.category,
            'suggested_fixes': self.suggested_fixes,
            'reproduction_steps': self.reproduction_steps,
            'recovery_actions': self.recovery_actions,
            'action_taken': self.action_taken,
            'action_result': self.action_result
        }

@dataclass
class ErrorPattern:
    """Pattern-based error recognition with recovery strategies"""
    pattern_id: str
    error_pattern: str
    category: str
    severity: str
    description: str
    common_causes: List[str]
    fix_suggestions: List[str]
    prevention_tips: List[str]
    recovery_actions: List[Dict[str, Any]] = field(default_factory=list)
    auto_recovery: bool = False

# ═══════════════════════════════════════════════════════════════════
# PRODUCTION-GRADE ERROR PINPOINTER WITH RECOVERY
# ═══════════════════════════════════════════════════════════════════

class ErrorPinpointer:
    """
    Advanced error analysis system with automated recovery.
    FIXED: Integration with orchestrator recovery mechanisms.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        """Initialize with comprehensive error tracking and recovery"""
        
        self.orchestrator = orchestrator
        self.logger = RotatingLogger("ErrorPinpointer", max_lines=10000)
        
        # ═══════════════════════════════════════════════════════════
        # Error Tracking & History
        # ═══════════════════════════════════════════════════════════
        self.error_history: deque = deque(maxlen=1000)
        self.error_patterns = defaultdict(int)
        self.module_error_counts = defaultdict(int)
        self.error_correlations = defaultdict(list)
        
        # Recovery tracking
        self.recovery_attempts = defaultdict(int)
        self.successful_recoveries = defaultdict(int)
        self.recovery_history: deque = deque(maxlen=500)
        
        # ═══════════════════════════════════════════════════════════
        # Built-in Error Patterns with Recovery Actions
        # ═══════════════════════════════════════════════════════════
        self.known_patterns = self._initialize_error_patterns()
        
        # ═══════════════════════════════════════════════════════════
        # Performance Tracking
        # ═══════════════════════════════════════════════════════════
        self.analysis_times = deque(maxlen=100)
        self.fix_success_rates = defaultdict(float)
        
        # Pattern matching cache for performance
        self._pattern_cache = {}
        self._cache_lock = threading.Lock()
        
        # Automatic recovery system
        self._recovery_executor = None
        self._recovery_queue = asyncio.Queue(maxsize=100)
        self._recovery_task = None
        
        self.logger.info("✅ ErrorPinpointer initialized with advanced debugging and recovery")
    
    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize known error patterns with recovery actions"""
        
        return [
            ErrorPattern(
                pattern_id="KEY_ERROR_INFOBUS",
                error_pattern=r"KeyError.*'([^']+)'",
                category="data",
                severity="high",
                description="Missing key in InfoBus data access",
                common_causes=[
                    "Module providing data hasn't executed yet",
                    "Data key name mismatch",
                    "Module dependency order issue",
                    "Conditional data provision not met"
                ],
                fix_suggestions=[
                    "Check module execution order in orchestrator",
                    "Verify data key spelling and case",
                    "Add default value handling",
                    "Check module dependency requirements"
                ],
                prevention_tips=[
                    "Use smart_bus.get(key, default=None)",
                    "Validate data availability before access",
                    "Implement proper module dependencies"
                ],
                recovery_actions=[
                    {
                        'action': 'request_missing_data',
                        'params': {'timeout': 5.0}
                    },
                    {
                        'action': 'reorder_execution',
                        'params': {'check_dependencies': True}
                    }
                ],
                auto_recovery=True
            ),
            
            ErrorPattern(
                pattern_id="TIMEOUT_MODULE",
                error_pattern=r"TimeoutError|timeout",
                category="timeout",
                severity="critical",
                description="Module execution timeout",
                common_causes=[
                    "Infinite loop in module logic",
                    "Blocking I/O operations",
                    "Deadlock in resource access",
                    "Complex computation taking too long"
                ],
                fix_suggestions=[
                    "Profile module execution time",
                    "Add progress checkpoints",
                    "Break large operations into chunks",
                    "Use async/await for I/O operations"
                ],
                prevention_tips=[
                    "Set reasonable timeout limits",
                    "Monitor execution time metrics",
                    "Use circuit breakers for external calls"
                ],
                recovery_actions=[
                    {
                        'action': 'disable_module_temporarily',
                        'params': {'duration': 60}
                    },
                    {
                        'action': 'reduce_module_load',
                        'params': {'factor': 0.5}
                    }
                ],
                auto_recovery=True
            ),
            
            ErrorPattern(
                pattern_id="MEMORY_ERROR",
                error_pattern=r"MemoryError|out of memory",
                category="resource",
                severity="critical",
                description="System running out of memory",
                common_causes=[
                    "Large dataset loading without batching",
                    "Memory leaks in module code",
                    "Accumulating historical data",
                    "Inefficient data structures"
                ],
                fix_suggestions=[
                    "Implement data batching",
                    "Add memory cleanup routines",
                    "Limit historical data retention",
                    "Use memory-efficient data structures"
                ],
                prevention_tips=[
                    "Monitor memory usage regularly",
                    "Implement garbage collection",
                    "Use streaming data processing"
                ],
                recovery_actions=[
                    {
                        'action': 'force_garbage_collection',
                        'params': {'generations': 2}
                    },
                    {
                        'action': 'clear_caches',
                        'params': {'preserve_critical': True}
                    },
                    {
                        'action': 'enter_emergency_mode',
                        'params': {'reason': 'Memory critical'}
                    }
                ],
                auto_recovery=True
            ),
            
            ErrorPattern(
                pattern_id="CIRCUIT_BREAKER_OPEN",
                error_pattern=r"Circuit breaker open|circuit.*broken",
                category="circuit",
                severity="high",
                description="Module circuit breaker activated",
                common_causes=[
                    "Repeated module failures",
                    "External service unavailable",
                    "Resource exhaustion",
                    "Configuration issues"
                ],
                fix_suggestions=[
                    "Check module logs for root cause",
                    "Verify external dependencies",
                    "Review module configuration",
                    "Test module in isolation"
                ],
                prevention_tips=[
                    "Implement proper error handling",
                    "Add retry logic with backoff",
                    "Monitor module health metrics"
                ],
                recovery_actions=[
                    {
                        'action': 'wait_and_reset_breaker',
                        'params': {'wait_time': 30}
                    },
                    {
                        'action': 'check_module_health',
                        'params': {'deep_check': True}
                    }
                ],
                auto_recovery=False
            ),
            
            ErrorPattern(
                pattern_id="NONE_TYPE_ERROR",
                error_pattern=r"NoneType.*has no attribute|AttributeError.*None",
                category="logic",
                severity="medium",
                description="Attempting to use None value",
                common_causes=[
                    "Function returning None unexpectedly",
                    "Uninitialized variable access",
                    "Failed data retrieval without error handling",
                    "Optional parameter not provided"
                ],
                fix_suggestions=[
                    "Add null checks before attribute access",
                    "Initialize variables with proper defaults",
                    "Handle function return values gracefully",
                    "Use Optional type hints"
                ],
                prevention_tips=[
                    "Use defensive programming practices",
                    "Validate inputs and outputs",
                    "Implement proper error handling"
                ],
                recovery_actions=[
                    {
                        'action': 'provide_default_value',
                        'params': {'use_last_known': True}
                    }
                ],
                auto_recovery=False
            ),
            
            ErrorPattern(
                pattern_id="CIRCULAR_DEPENDENCY",
                error_pattern=r"circular.*dependency|maximum recursion|RecursionError",
                category="dependency",
                severity="critical",
                description="Circular dependency in module system",
                common_causes=[
                    "Module A requires output from Module B which requires Module A",
                    "Recursive data dependencies",
                    "Improper module initialization order"
                ],
                fix_suggestions=[
                    "Redesign module dependencies",
                    "Introduce intermediate data layers",
                    "Break dependency cycles with default values",
                    "Use dependency injection patterns"
                ],
                prevention_tips=[
                    "Design clear data flow architecture",
                    "Validate dependency graph at startup",
                    "Use topological sorting for execution order"
                ],
                recovery_actions=[
                    {
                        'action': 'break_dependency_cycle',
                        'params': {'method': 'remove_weakest'}
                    },
                    {
                        'action': 'rebuild_execution_plan',
                        'params': {'validate': True}
                    }
                ],
                auto_recovery=True
            )
        ]
    
    def analyze_error(self, exception: Exception, module_name: str = "Unknown") -> ErrorContext:
        """
        Comprehensive error analysis with automated recovery.
        ENHANCED: Integration with orchestrator recovery mechanisms.
        """
        
        start_time = time.time()
        
        try:
            # ═══════════════════════════════════════════════════════════
            # Extract Basic Error Information
            # ═══════════════════════════════════════════════════════════
            
            error_type = type(exception).__name__
            error_message = str(exception)
            
            # Get traceback information
            tb = exception.__traceback__
            if tb:
                frame = tb.tb_frame
                file_path = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = tb.tb_lineno
            else:
                frame = inspect.currentframe()
                if frame is not None:
                    file_path = frame.f_code.co_filename
                    function_name = frame.f_code.co_name
                    line_number = frame.f_lineno
                else:
                    file_path = "unknown"
                    function_name = "unknown"
                    line_number = 0
            
            # ═══════════════════════════════════════════════════════════
            # Create Error Context
            # ═══════════════════════════════════════════════════════════
            
            context = ErrorContext(
                error_type=error_type,
                error_message=error_message,
                module_name=module_name,
                function_name=function_name,
                file_path=file_path,
                line_number=line_number,
                timestamp=datetime.now()
            )
            
            # ═══════════════════════════════════════════════════════════
            # Extract Code Context
            # ═══════════════════════════════════════════════════════════
            
            context.source_lines = self._extract_source_lines(file_path, line_number)
            context.local_variables = self._extract_local_variables(frame)
            context.call_stack = self._extract_call_stack(tb)
            
            # ═══════════════════════════════════════════════════════════
            # Extract System Context
            # ═══════════════════════════════════════════════════════════
            
            context.module_state = self._get_module_state(module_name)
            context.infobus_snapshot = self._get_infobus_snapshot()
            context.related_errors = self._find_related_errors(error_type, module_name)
            
            # ═══════════════════════════════════════════════════════════
            # Analyze Error Pattern and Generate Recovery Actions
            # ═══════════════════════════════════════════════════════════
            
            self._analyze_error_pattern(context)
            context.recovery_actions = self._generate_recovery_actions(context)
            
            # ═══════════════════════════════════════════════════════════
            # Generate Debugging Steps
            # ═══════════════════════════════════════════════════════════
            
            context.suggested_fixes = self._generate_fix_suggestions(context)
            context.reproduction_steps = self._generate_reproduction_steps(context)
            
            # ═══════════════════════════════════════════════════════════
            # Record Error and Correlate
            # ═══════════════════════════════════════════════════════════
            
            self.error_history.append(context)
            self.error_patterns[f"{error_type}:{module_name}"] += 1
            self.module_error_counts[module_name] += 1
            self._correlate_error(context)
            
            # ═══════════════════════════════════════════════════════════
            # Attempt Automatic Recovery if Enabled
            # ═══════════════════════════════════════════════════════════
            
            if context.recovery_actions and self._should_attempt_recovery(context):
                asyncio.create_task(self._attempt_recovery(context))
            
            # ═══════════════════════════════════════════════════════════
            # Performance Tracking
            # ═══════════════════════════════════════════════════════════
            
            analysis_time = (time.time() - start_time) * 1000
            self.analysis_times.append(analysis_time)
            
            # ═══════════════════════════════════════════════════════════
            # Logging
            # ═══════════════════════════════════════════════════════════
            
            self.logger.error(format_operator_message(
                "💥",
                message=f"ERROR ANALYZED: {error_type} in {module_name}::{function_name}:{line_number}",
                severity=context.severity,
                category=context.category,
                recovery_planned=len(context.recovery_actions) > 0
            ))
            
            return context
            
        except Exception as analysis_error:
            # Fallback error context if analysis fails
            self.logger.error(f"Error analysis failed: {analysis_error}")
            
            return ErrorContext(
                error_type=type(exception).__name__,
                error_message=str(exception),
                module_name=module_name,
                function_name="unknown",
                file_path="unknown",
                line_number=0,
                timestamp=datetime.now(),
                severity="high",
                category="analysis_failed",
                suggested_fixes=["Manual debugging required - error analysis failed"]
            )
    
    def _correlate_error(self, context: ErrorContext):
        """Correlate error with recent system events"""
        correlation_key = f"{context.error_type}:{context.module_name}"
        
        # Find temporal correlations
        recent_errors = list(self.error_history)[-20:]
        correlated = []
        
        for error in recent_errors:
            if error == context:
                continue
            
            # Check temporal proximity (within 5 seconds)
            time_diff = abs((context.timestamp - error.timestamp).total_seconds())
            if time_diff < 5:
                correlated.append({
                    'error': f"{error.error_type} in {error.module_name}",
                    'time_diff': time_diff,
                    'severity': error.severity
                })
        
        if correlated:
            self.error_correlations[correlation_key].extend(correlated)
            context.related_errors.extend([c['error'] for c in correlated[:3]])
    
    def _should_attempt_recovery(self, context: ErrorContext) -> bool:
        """Determine if automatic recovery should be attempted"""
        # Don't attempt if too many recent failures
        recovery_key = f"{context.module_name}:{context.error_type}"
        
        if self.recovery_attempts[recovery_key] > 5:
            success_rate = self.successful_recoveries[recovery_key] / self.recovery_attempts[recovery_key]
            if success_rate < 0.2:  # Less than 20% success rate
                return False
        
        # Check if pattern allows auto-recovery
        for pattern in self.known_patterns:
            if re.search(pattern.error_pattern, context.error_message, re.IGNORECASE):
                return pattern.auto_recovery
        
        return False
    
    async def _attempt_recovery(self, context: ErrorContext):
        """Attempt automatic recovery based on error context"""
        recovery_key = f"{context.module_name}:{context.error_type}"
        self.recovery_attempts[recovery_key] += 1
        
        try:
            for action in context.recovery_actions:
                action_type = action.get('action')
                params = action.get('params', {})
                if not action_type:
                    continue  # Skip if action_type is None
                self.logger.info(f"🔧 Attempting recovery action: {action_type} for {context.module_name}")
                
                success = await self._execute_recovery_action(action_type, params, context)
                
                if success:
                    self.successful_recoveries[recovery_key] += 1
                    context.action_taken = True
                    context.action_result = f"Recovery successful: {action_type}"
                    
                    self.recovery_history.append({
                        'timestamp': datetime.now(),
                        'module': context.module_name,
                        'error_type': context.error_type,
                        'action': action_type,
                        'success': True
                    })
                    
                    self.logger.info(f"✅ Recovery successful: {action_type} for {context.module_name}")
                    break
                    
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            context.action_result = f"Recovery failed: {str(recovery_error)}"
    
    async def _execute_recovery_action(self, action_type: str, params: Dict[str, Any], context: ErrorContext) -> bool:
        """Execute specific recovery action"""
        
        if not self.orchestrator:
            return False
        
        try:
            if action_type == 'request_missing_data':
                # Request missing data from InfoBus
                key_match = re.search(r"['\"]([^'\"]+)['\"]", context.error_message)
                if key_match:
                    key = key_match.group(1)
                    smart_bus = InfoBusManager.get_instance()
                    smart_bus.request_data(key, context.module_name)
                    
                    # Wait for data
                    timeout = params.get('timeout', 5.0)
                    await asyncio.sleep(timeout)
                    
                    # Check if data is now available
                    if smart_bus.get(key, context.module_name) is not None:
                        return True
            
            elif action_type == 'disable_module_temporarily':
                # Temporarily disable problematic module
                duration = params.get('duration', 60)
                self.orchestrator.disable_module(context.module_name)
                
                # Schedule re-enable
                async def re_enable():
                    await asyncio.sleep(duration)
                    if self.orchestrator:
                        self.orchestrator.enable_module(context.module_name)
                
                asyncio.create_task(re_enable())
                return True
            
            elif action_type == 'force_garbage_collection':
                # Force garbage collection
                import gc
                generations = params.get('generations', 2)
                gc.collect(generations)
                
                # Clear some caches
                if params.get('preserve_critical', True):
                    smart_bus = InfoBusManager.get_instance()
                    smart_bus._cleanup_old_data()
                
                return True
            
            elif action_type == 'enter_emergency_mode':
                # Trigger emergency mode
                reason = params.get('reason', f"Error in {context.module_name}")
                self.orchestrator.trigger_emergency_mode_manually(reason)
                return True
            
            elif action_type == 'wait_and_reset_breaker':
                # Wait and reset circuit breaker
                wait_time = params.get('wait_time', 30)
                await asyncio.sleep(wait_time)
                
                return self.orchestrator.reset_circuit_breaker(context.module_name)
            
            elif action_type == 'break_dependency_cycle':
                # Rebuild execution plan
                self.orchestrator.build_execution_plan()
                return True
            
            elif action_type == 'reduce_module_load':
                # Reduce module load by adjusting config
                factor = params.get('factor', 0.5)
                
                if hasattr(self.orchestrator.modules.get(context.module_name), 'reduce_load'):
                    module = self.orchestrator.modules[context.module_name]
                    module.reduce_load(factor)
                    return True
                    
        except Exception as e:
            self.logger.error(f"Recovery action {action_type} failed: {e}")
        
        return False
    
    def _generate_recovery_actions(self, context: ErrorContext) -> List[Dict[str, Any]]:
        """Generate recovery actions based on error pattern"""
        actions = []
        
        # Check known patterns
        for pattern in self.known_patterns:
            if re.search(pattern.error_pattern, context.error_message, re.IGNORECASE):
                actions.extend(pattern.recovery_actions)
                break
        
        # Add generic recovery actions based on severity
        if not actions:
            if context.severity == "critical":
                actions.append({
                    'action': 'enter_emergency_mode',
                    'params': {'reason': f"Critical error in {context.module_name}"}
                })
            elif context.severity == "high":
                actions.append({
                    'action': 'disable_module_temporarily',
                    'params': {'duration': 30}
                })
        
        return actions
    
    def _analyze_error_pattern(self, context: ErrorContext):
        """Analyze error against known patterns with caching"""
        # Check cache first
        cache_key = f"{context.error_type}:{context.error_message[:50]}"
        
        with self._cache_lock:
            if cache_key in self._pattern_cache:
                cached = self._pattern_cache[cache_key]
                context.category = cached['category']
                context.severity = cached['severity']
                context.suggested_fixes.extend(cached['fixes'])
                return
        
        # Pattern matching
        for pattern in self.known_patterns:
            if re.search(pattern.error_pattern, context.error_message, re.IGNORECASE):
                context.category = pattern.category
                context.severity = pattern.severity
                context.suggested_fixes.extend(pattern.fix_suggestions)
                
                # Cache result
                with self._cache_lock:
                    self._pattern_cache[cache_key] = {
                        'category': pattern.category,
                        'severity': pattern.severity,
                        'fixes': pattern.fix_suggestions
                    }
                return
        
        # Fallback classification
        if "timeout" in context.error_message.lower():
            context.category = "timeout"
            context.severity = "high"
        elif "memory" in context.error_message.lower():
            context.category = "resource"
            context.severity = "critical"
        elif context.error_type in ["KeyError", "AttributeError"]:
            context.category = "data"
            context.severity = "medium"
        else:
            context.category = "logic"
            context.severity = "medium"
    
    def correlate_errors(self, timeframe_minutes: int = 10) -> List[ErrorPattern]:
        """
        Identify error patterns and correlations across system.
        ENHANCED: Returns actionable patterns for orchestrator.
        """
        patterns = []
        cutoff_time = datetime.now().timestamp() - (timeframe_minutes * 60)
        
        # Group errors by type and module
        error_groups = defaultdict(list)
        
        for error in self.error_history:
            if error.timestamp.timestamp() < cutoff_time:
                continue
            
            key = f"{error.error_type}:{error.module_name}"
            error_groups[key].append(error)
        
        # Analyze patterns
        for key, errors in error_groups.items():
            if len(errors) < 2:
                continue
            
            # Calculate frequency
            time_span = (errors[-1].timestamp - errors[0].timestamp).total_seconds()
            frequency = len(errors) / max(time_span / 60, 1)  # errors per minute
            
            # Find common factors
            common_dependencies = set.intersection(*[set(e.module_state.get('dependencies', [])) for e in errors])
            
            # Create pattern report
            pattern_report = ErrorPattern(
                pattern_id=f"dynamic_{key}_{int(time.time())}",
                error_pattern=key,
                category=errors[0].category,
                severity="critical" if frequency > 1 else "high",
                description=f"Recurring {errors[0].error_type} in {errors[0].module_name}",
                common_causes=[f"Frequency: {frequency:.2f} errors/minute"],
                fix_suggestions=[
                    f"Review module {errors[0].module_name} implementation",
                    "Check system resources",
                    "Verify dependencies"
                ],
                prevention_tips=[
                    "Add better error handling",
                    "Implement circuit breaker",
                    "Add input validation"
                ],
                recovery_actions=[
                    {
                        'action': 'disable_module_temporarily',
                        'params': {'duration': 300}
                    }
                ],
                auto_recovery=True
            )
            
            patterns.append(pattern_report)
        
        # Store patterns for orchestrator
        if self.orchestrator and patterns:
            smart_bus = InfoBusManager.get_instance()
            smart_bus.set(
                'error_patterns',
                {
                    'timestamp': datetime.now().isoformat(),
                    'patterns': [p.__dict__ for p in patterns],
                    'action_required': len(patterns) > 5
                },
                module='ErrorPinpointer',
                thesis=f"Detected {len(patterns)} error patterns requiring attention"
            )
        
        return patterns
    
    def create_debugging_guide(self, context: ErrorContext) -> str:
        """
        Generate actionable debugging guide.
        ENHANCED: Integration with system state.
        """
        guide = f"""
🔍 DEBUGGING GUIDE: {context.error_type}
{'=' * 60}

📍 ERROR LOCATION:
  Module: {context.module_name}
  Function: {context.function_name}
  File: {Path(context.file_path).name}
  Line: {context.line_number}

⚠️ ERROR DETAILS:
  Type: {context.error_type}
  Message: {context.error_message}
  Severity: {context.severity.upper()}
  Category: {context.category}

📊 SYSTEM STATE:
  Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
  Module Errors: {self.module_error_counts.get(context.module_name, 0)}
  Related Errors: {len(context.related_errors)}

💡 SUGGESTED FIXES:
"""
        
        for i, fix in enumerate(context.suggested_fixes, 1):
            guide += f"  {i}. {fix}\n"
        
        guide += "\n🔄 REPRODUCTION STEPS:\n"
        for i, step in enumerate(context.reproduction_steps, 1):
            guide += f"  {i}. {step}\n"
        
        # Add recovery actions if available
        if context.recovery_actions:
            guide += "\n🔧 AUTOMATED RECOVERY:\n"
            for action in context.recovery_actions:
                guide += f"  • {action['action']} "
                if action.get('params'):
                    guide += f"({', '.join(f'{k}={v}' for k, v in action['params'].items())})"
                guide += "\n"
        
        # Add code context
        if context.source_lines:
            guide += "\n📝 CODE CONTEXT:\n"
            guide += "```python\n"
            guide += "\n".join(context.source_lines[:10])
            guide += "\n```\n"
        
        # Add local variables if available
        if context.local_variables:
            guide += "\n🔢 LOCAL VARIABLES:\n"
            for var, value in list(context.local_variables.items())[:5]:
                guide += f"  {var} = {value}\n"
        
        # Add orchestrator integration
        if self.orchestrator:
            guide += "\n🚦 ORCHESTRATOR STATUS:\n"
            emergency_status = self.orchestrator.get_emergency_mode_status()
            if emergency_status['active']:
                guide += f"  ⚠️ EMERGENCY MODE ACTIVE: {emergency_status['reason']}\n"
            
            cb_status = self.orchestrator.get_circuit_breaker_status()
            if context.module_name in cb_status:
                cb = cb_status[context.module_name]
                guide += f"  Circuit Breaker: {cb['state']} (failures: {cb['failure_count']})\n"
        
        guide += "\n💻 QUICK ACTIONS:\n"
        guide += "  1. Check module logs: error_pinpointer.get_module_error_log(module_name)\n"
        guide += "  2. View error history: error_pinpointer.get_error_summary()\n"
        guide += "  3. Check correlations: error_pinpointer.correlate_errors()\n"
        
        if self.orchestrator:
            guide += "  4. Reset module: orchestrator.reset_circuit_breaker(module_name)\n"
            guide += "  5. Disable module: orchestrator.disable_module(module_name)\n"
        
        return guide
    
    def _extract_source_lines(self, file_path: str, line_number: int, context_lines: int = 10) -> List[str]:
        """Extract source code lines around the error"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            context_lines_data = []
            for i in range(start, end):
                marker = ">>> " if i == line_number - 1 else "    "
                context_lines_data.append(f"{marker}{i+1:4d}: {lines[i].rstrip()}")
            
            return context_lines_data
            
        except Exception as e:
            return [f"Could not read source file: {e}"]
    
    def _extract_local_variables(self, frame) -> Dict[str, Any]:
        """Extract local variables with safe serialization"""
        
        local_vars = {}
        
        if not frame:
            return local_vars
        
        for var_name, var_value in frame.f_locals.items():
            if var_name.startswith('__'):
                continue
                
            try:
                # Limit size of stored values
                if isinstance(var_value, (str, bytes)):
                    if len(var_value) > 1000:
                        local_vars[var_name] = f"{type(var_value).__name__}(length={len(var_value)})"
                    else:
                        local_vars[var_name] = var_value
                elif isinstance(var_value, (list, dict, set)):
                    local_vars[var_name] = f"{type(var_value).__name__}(length={len(var_value)})"
                elif isinstance(var_value, (int, float, bool, type(None))):
                    local_vars[var_name] = var_value
                elif isinstance(var_value, np.ndarray):
                    local_vars[var_name] = f"ndarray(shape={var_value.shape}, dtype={var_value.dtype})"
                else:
                    local_vars[var_name] = f"{type(var_value).__name__}"
            except:
                local_vars[var_name] = "<Unable to serialize>"
        
        return local_vars
    
    def _extract_call_stack(self, tb) -> List[Dict[str, Any]]:
        """Extract complete call stack information"""
        
        stack = []
        
        while tb:
            frame = tb.tb_frame
            stack.append({
                'file': frame.f_code.co_filename,
                'function': frame.f_code.co_name,
                'line': tb.tb_lineno,
                'module': frame.f_globals.get('__name__', 'unknown')
            })
            tb = tb.tb_next
        
        return stack
    
    def _get_module_state(self, module_name: str) -> Dict[str, Any]:
        """Get current module state if available"""
        
        if not self.orchestrator:
            return {}
        
        try:
            module = self.orchestrator.get_module_by_name(module_name)
            if module and hasattr(module, 'get_state'):
                state = module.get_state()
                
                # Add orchestrator context
                cb_status = self.orchestrator.get_circuit_breaker_status()
                if module_name in cb_status:
                    state['circuit_breaker'] = cb_status[module_name]
                
                # Add performance metrics
                if module_name in self.orchestrator.module_performance:
                    state['performance'] = self.orchestrator.module_performance[module_name]
                
                return state
        except:
            pass
        
        return {"error": "Failed to get module state"}
    
    def _get_infobus_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current InfoBus state"""
        
        try:
            smart_bus = InfoBusManager.get_instance()
            
            snapshot = {
                'data_keys': list(smart_bus._data_store.keys())[:20],  # First 20 keys
                'total_keys': len(smart_bus._data_store),
                'disabled_modules': list(smart_bus._module_disabled),
                'recent_events': list(smart_bus._event_log)[-10:] if hasattr(smart_bus, '_event_log') else [],
                'performance_metrics': smart_bus.get_performance_metrics()
            }
            
            # Add memory usage
            try:
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                snapshot['memory_usage_mb'] = round(memory_usage, 2)
            except:
                pass
            
            return snapshot
            
        except Exception as e:
            return {"error": f"Failed to get InfoBus snapshot: {e}"}
    
    def _find_related_errors(self, error_type: str, module_name: str, lookback_minutes: int = 10) -> List[str]:
        """Find related errors in recent history"""
        
        related = []
        cutoff_time = datetime.now().timestamp() - (lookback_minutes * 60)
        
        for error in self.error_history:
            if error.timestamp.timestamp() < cutoff_time:
                continue
            
            # Same error type in different modules
            if error.error_type == error_type and error.module_name != module_name:
                related.append(f"Same error in {error.module_name}")
            
            # Different error in same module
            elif error.module_name == module_name and error.error_type != error_type:
                related.append(f"{error.error_type} in same module")
            
            # Check for cascade effects
            if error.severity == "critical" and abs((error.timestamp - datetime.now()).total_seconds()) < 30:
                related.append(f"Critical error: {error.error_type} in {error.module_name}")
        
        return related[:5]  # Limit to 5 most recent
    
    def _generate_fix_suggestions(self, context: ErrorContext) -> List[str]:
        """Generate specific fix suggestions based on error context"""
        
        suggestions = []
        
        # Add suggestions from pattern if already present
        if context.suggested_fixes:
            return context.suggested_fixes
        
        # Generate based on error type
        suggestions.append(f"1. Check line {context.line_number} in {Path(context.file_path).name}")
        
        # Error-specific suggestions
        if "KeyError" in context.error_type:
            key_match = re.search(r"['\"]([^'\"]+)['\"]", context.error_message)
            if key_match:
                key = key_match.group(1)
                suggestions.extend([
                    f"2. Verify that '{key}' is being set in InfoBus",
                    f"3. Check module that should provide '{key}'",
                    f"4. Use smart_bus.get('{key}', default_value) instead",
                    "5. Add dependency check before accessing data"
                ])
        
        elif "timeout" in context.error_message.lower():
            suggestions.extend([
                "2. Profile the module to find slow operations",
                "3. Check for infinite loops or recursive calls",
                "4. Consider breaking operation into smaller chunks",
                "5. Increase timeout or optimize algorithm"
            ])
        
        elif "NoneType" in context.error_message:
            suggestions.extend([
                "2. Add null checks before the error line",
                "3. Trace back to find where None is coming from",
                "4. Set appropriate default values",
                "5. Use Optional type hints for clarity"
            ])
        
        elif "memory" in context.error_message.lower():
            suggestions.extend([
                "2. Check for memory leaks in loops",
                "3. Implement data batching",
                "4. Clear unused variables and caches",
                "5. Use generators instead of lists for large data"
            ])
        
        # Add recovery suggestion
        if context.recovery_actions:
            suggestions.append(f"{len(suggestions)+1}. Automatic recovery available - monitor results")
        
        return suggestions
    
    def _generate_reproduction_steps(self, context: ErrorContext) -> List[str]:
        """Generate steps to reproduce the error"""
        
        steps = [
            f"1. Load module: {context.module_name}",
            f"2. Execute function: {context.function_name}",
            f"3. Monitor line: {context.line_number}",
        ]
        
        # Add local variable setup if available
        if context.local_variables:
            steps.append("4. Set local variables:")
            for var, value in list(context.local_variables.items())[:3]:
                steps.append(f"   {var} = {value}")
        
        # Add InfoBus state if relevant
        if "KeyError" in context.error_type:
            steps.append("5. Check InfoBus state for missing keys")
        
        # Add system state
        if context.module_state.get('circuit_breaker'):
            cb = context.module_state['circuit_breaker']
            steps.append(f"6. Circuit breaker state: {cb['state']}")
        
        return steps
    
    # ═══════════════════════════════════════════════════════════════
    # Analysis & Reporting Methods
    # ═══════════════════════════════════════════════════════════════
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary"""
        
        summary = {
            'total_errors': len(self.error_history),
            'unique_modules': len(set(e.module_name for e in self.error_history)),
            'most_common_errors': [],
            'problem_modules': [],
            'error_trends': {},
            'avg_analysis_time_ms': np.mean(self.analysis_times) if self.analysis_times else 0,
            'recovery_stats': {
                'total_attempts': sum(self.recovery_attempts.values()),
                'successful': sum(self.successful_recoveries.values()),
                'success_rate': sum(self.successful_recoveries.values()) / max(sum(self.recovery_attempts.values()), 1)
            }
        }
        
        # Most common error types
        error_counts = {}
        module_errors = {}
        severity_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        for error in self.error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            module_errors[error.module_name] = module_errors.get(error.module_name, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        
        # Sort by frequency
        summary['most_common_errors'] = sorted(
            error_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        summary['problem_modules'] = sorted(
            module_errors.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        summary['severity_distribution'] = severity_counts
        
        # Error trends (last hour)
        hour_ago = datetime.now().timestamp() - 3600
        recent_errors = [e for e in self.error_history if e.timestamp.timestamp() > hour_ago]
        
        if recent_errors:
            # Group by 5-minute intervals
            intervals = defaultdict(int)
            for error in recent_errors:
                interval = int(error.timestamp.timestamp() / 300) * 300
                intervals[interval] += 1
            
            summary['error_trends'] = dict(sorted(intervals.items()))
        
        return summary
    
    def get_module_error_log(self, module_name: str) -> List[Dict[str, Any]]:
        """Get detailed error log for specific module"""
        
        module_errors = []
        
        for error in self.error_history:
            if error.module_name == module_name:
                module_errors.append({
                    'timestamp': error.timestamp.isoformat(),
                    'error_type': error.error_type,
                    'message': error.error_message[:100] + '...' if len(error.error_message) > 100 else error.error_message,
                    'severity': error.severity,
                    'category': error.category,
                    'line': error.line_number,
                    'function': error.function_name,
                    'recovery_attempted': error.action_taken,
                    'recovery_result': error.action_result
                })
        
        return module_errors
    
    def create_debug_snapshot(self, module_name: str) -> Dict[str, Any]:
        """Create comprehensive debugging snapshot for a module"""
        
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'recent_errors': self.get_module_error_log(module_name)[-10:],
            'module_state': self._get_module_state(module_name),
            'infobus_state': self._get_infobus_snapshot(),
            'execution_context': {},
            'recovery_history': []
        }
        
        # Add execution context from orchestrator
        if self.orchestrator:
            smart_bus = InfoBusManager.get_instance()
            snapshot['execution_context'] = {
                'is_enabled': smart_bus.is_module_enabled(module_name),
                'failure_count': smart_bus._circuit_breakers[module_name].failure_count if module_name in smart_bus._circuit_breakers else 0,
                'circuit_breaker': self.orchestrator.circuit_breakers.get(module_name, {})
            }
            
            # Get performance metrics
            if module_name in self.orchestrator.module_performance:
                snapshot['performance'] = self.orchestrator.module_performance[module_name]
        
        # Add recovery history
        for record in self.recovery_history:
            if record['module'] == module_name:
                snapshot['recovery_history'].append(record)
        
        return snapshot
    
    def export_error_report(self, filepath: str, last_n_errors: int = 100):
        """Export comprehensive error report"""
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_errors_tracked': len(self.error_history),
                'analysis_period': f'Last {last_n_errors} errors',
                'avg_analysis_time_ms': np.mean(self.analysis_times) if self.analysis_times else 0,
                'system_info': {
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent(interval=0.1)
                }
            },
            'summary': self.get_error_summary(),
            'detailed_errors': [
                error.to_dict() 
                for error in list(self.error_history)[-last_n_errors:]
            ],
            'patterns': dict(self.error_patterns),
            'module_statistics': dict(self.module_error_counts),
            'correlations': dict(self.error_correlations),
            'recovery_statistics': {
                'attempts': dict(self.recovery_attempts),
                'successes': dict(self.successful_recoveries),
                'recent_recoveries': list(self.recovery_history)[-20:]
            }
        }
        
        # Add orchestrator state if available
        if self.orchestrator:
            report['orchestrator_state'] = {
                'emergency_mode': self.orchestrator.get_emergency_mode_status(),
                'circuit_breakers': self.orchestrator.get_circuit_breaker_status()
            }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📊 Error report exported to {filepath}")
    
    def get_debugging_guide(self, error_type: Optional[str] = None, module_name: Optional[str] = None) -> str:
        """Generate comprehensive debugging guide"""
        
        guide = f"""
🔍 SMARTINFOBUS DEBUGGING GUIDE
{'=' * 50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target: {error_type or 'All'} errors in {module_name or 'All'} modules

📋 QUICK DEBUGGING CHECKLIST
{'─' * 30}
1. ✅ Check module execution order
2. ✅ Verify InfoBus data availability  
3. ✅ Validate input parameters
4. ✅ Review error context and local variables
5. ✅ Check for related errors in other modules
6. ✅ Examine system resource usage
7. ✅ Verify circuit breaker states
8. ✅ Check for emergency mode activation

🛠️ COMMON ERROR PATTERNS
{'─' * 30}
"""
        
        # Add relevant patterns
        for pattern in self.known_patterns:
            if not error_type or pattern.error_pattern.lower() in error_type.lower():
                guide += f"""
{pattern.pattern_id}:
  Type: {pattern.category} ({pattern.severity})
  Description: {pattern.description}
  Common Causes: {', '.join(pattern.common_causes[:2])}
  Quick Fix: {pattern.fix_suggestions[0] if pattern.fix_suggestions else 'See documentation'}
  Auto-Recovery: {'Yes' if pattern.auto_recovery else 'No'}
"""
        
        # Add module-specific insights
        if module_name and module_name in self.module_error_counts:
            error_count = self.module_error_counts[module_name]
            guide += f"""

🎯 MODULE-SPECIFIC INSIGHTS: {module_name}
{'─' * 30}
- Total errors: {error_count}
- Most common: {self._get_most_common_error_for_module(module_name)}
- Status: {'⚠️ High error rate' if error_count > 10 else '✅ Normal'}
- Recovery success rate: {self._get_recovery_rate_for_module(module_name):.1%}
"""
        
        # Add system status
        if self.orchestrator:
            emergency_status = self.orchestrator.get_emergency_mode_status()
            guide += f"""

🚦 SYSTEM STATUS
{'─' * 30}
- Emergency Mode: {'ACTIVE' if emergency_status['active'] else 'Inactive'}
- Circuit Breakers Open: {sum(1 for cb in self.orchestrator.circuit_breakers.values() if cb.state == 'OPEN')}
- Total Modules: {len(self.orchestrator.modules)}
"""
        
        guide += """

🚀 ADVANCED DEBUGGING TOOLS
─────────────────────────────
1. error_pinpointer.create_debug_snapshot(module_name)
2. error_pinpointer.correlate_errors(timeframe_minutes)
3. error_pinpointer.export_error_report(filepath)
4. orchestrator.get_system_status_report()
5. smart_bus.get_performance_metrics()

🔧 RECOVERY TOOLS
─────────────────
1. orchestrator.reset_circuit_breaker(module_name)
2. orchestrator.exit_emergency_mode()
3. orchestrator.enable_module(module_name)
4. smart_bus.cleanup_old_data()

📞 NEED HELP?
─────────────
- Check SmartInfoBus documentation
- Review module dependency graph
- Use replay engine for step-by-step debugging
- Enable verbose logging for detailed traces
"""
        
        return guide
    
    def _get_most_common_error_for_module(self, module_name: str) -> str:
        """Get most common error type for a specific module"""
        
        module_errors = defaultdict(int)
        for error in self.error_history:
            if error.module_name == module_name:
                module_errors[error.error_type] += 1
        
        if module_errors:
            return max(module_errors.items(), key=lambda x: x[1])[0]
        return "No errors recorded"
    
    def _get_recovery_rate_for_module(self, module_name: str) -> float:
        """Get recovery success rate for module"""
        
        attempts = 0
        successes = 0
        
        for key, count in self.recovery_attempts.items():
            if key.startswith(f"{module_name}:"):
                attempts += count
                successes += self.successful_recoveries.get(key, 0)
        
        return successes / max(attempts, 1)
    
    def clear_history(self, keep_last_n: int = 100):
        """Clear error history, keeping only recent errors"""
        
        if len(self.error_history) > keep_last_n:
            recent_errors = list(self.error_history)[-keep_last_n:]
            self.error_history = deque(recent_errors, maxlen=1000)
            
            # Clear pattern cache
            with self._cache_lock:
                self._pattern_cache.clear()
            
            self.logger.info(f"🧹 Cleared error history, kept last {keep_last_n} errors")

# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def analyze_exception(exception: Exception, module_name: str = "Unknown") -> ErrorContext:
    """Quick error analysis function"""
    
    pinpointer = ErrorPinpointer()
    return pinpointer.analyze_error(exception, module_name)

def create_error_handler(module_name: str, pinpointer: Optional[ErrorPinpointer] = None):
    """Create error handler decorator for modules"""
    
    if not pinpointer:
        pinpointer = ErrorPinpointer()
    
    def error_handler(func):
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = pinpointer.analyze_error(e, module_name)
                
                # Log detailed error information
                guide = pinpointer.create_debugging_guide(context)
                print(guide)
                
                # Re-raise with enhanced context
                raise
        
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = pinpointer.analyze_error(e, module_name)
                
                # Log detailed error information
                guide = pinpointer.create_debugging_guide(context)
                print(guide)
                
                # Re-raise with enhanced context
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return error_handler