# ─────────────────────────────────────────────────────────────
# File: modules/core/error_pinpointer.py
# [ROCKET] PRODUCTION-READY Error Analysis & Debugging System
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# 2025 ENHANCEMENTS:
#   - Loop-safe async recovery worker (idempotent start/stop)
#   - NumPy-optional stats (robust without np)
#   - Pylance-friendly InfoBus/orchestrator duck-typing
#   - Rate-limited error deduplication (true lightweight fast-path)
#   - Pluggable recovery actions; compiled-regex cache
#   - Safe snapshots (bounded), hardened export utilities
#   - Decorators preserve metadata; sync/async supported
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import inspect
import json
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, TYPE_CHECKING

try:
    import numpy as np  # type: ignore
    NUMPY_AVAILABLE = True
except Exception:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    import psutil  # type: ignore
    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False

from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager  # type: ignore

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# CONFIG & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ErrorPinpointerConfig:
    max_history: int = 1000
    max_recovery_history: int = 500
    dedupe_window_sec: float = 3.0
    max_recovery_attempts_per_key: int = 5
    min_success_ratio_for_retries: float = 0.2
    recovery_queue_size: int = 100
    recovery_workers: int = 2
    snapshot_max_locals: int = 50
    snapshot_max_source_context: int = 10


@dataclass
class ErrorContext:
    """
    Comprehensive error context for precise debugging.
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
    severity: str = "unknown"  # critical, high, medium, low, duplicate
    category: str = "unknown"  # logic, data, timeout, dependency, resource
    suggested_fixes: List[str] = field(default_factory=list)
    reproduction_steps: List[str] = field(default_factory=list)

    # Recovery actions
    recovery_actions: List[Dict[str, Any]] = field(default_factory=list)
    action_taken: bool = False
    action_result: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
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
    """Pattern-based error recognition with recovery strategies."""
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
# UTIL STATS (NumPy optional)
# ═══════════════════════════════════════════════════════════════════

def _mean(values: List[float]) -> float:
    if not values:
        return 0.0
    if NUMPY_AVAILABLE and np is not None:
        try:
            return float(np.mean(values))  # type: ignore[call-arg]
        except Exception:
            pass
    return sum(values) / len(values)


# ═══════════════════════════════════════════════════════════════════
# ERROR PINPOINTER
# ═══════════════════════════════════════════════════════════════════

class ErrorPinpointer:
    """
    Advanced error analysis system with automated recovery.
    Loop-safe async worker, robust duck-typing, and rate-limited logging.
    """

    # -------- Recovery Action Registry (pluggable) --------
    _custom_recovery_actions: Dict[str, Callable[['ErrorPinpointer', Dict[str, Any], ErrorContext], Any]] = {}

    def __init__(self, orchestrator: Optional['ModuleOrchestrator'] = None, config: Optional[ErrorPinpointerConfig] = None):
        self.orchestrator = orchestrator
        self.cfg = config or ErrorPinpointerConfig()
        self.logger = RotatingLogger("ErrorPinpointer", log_path="logs/errors/error_pinpointer.log", max_lines=10000)

        # Thread-safety for shared structures
        self._history_lock = threading.RLock()
        self._cache_lock = threading.RLock()

        # Error Tracking & History
        self.error_history: Deque[ErrorContext] = deque(maxlen=self.cfg.max_history)
        self.error_patterns = defaultdict(int)
        self.module_error_counts = defaultdict(int)
        self.error_correlations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # Deduplication (rate-limiting)
        self._last_seen: Dict[str, float] = {}  # key -> last timestamp

        # Recovery tracking
        self.recovery_attempts = defaultdict(int)
        self.successful_recoveries = defaultdict(int)
        self.recovery_history: Deque[Dict[str, Any]] = deque(maxlen=self.cfg.max_recovery_history)

        # Built-in Error Patterns with Recovery Actions
        self.known_patterns = self._initialize_error_patterns()

        # Performance Tracking
        self.analysis_times: Deque[float] = deque(maxlen=100)

        # Pattern matching cache for performance
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._compiled_regex: Dict[str, re.Pattern] = {
            p.pattern_id: re.compile(p.error_pattern, re.IGNORECASE) for p in self.known_patterns
        }

        # Async recovery infrastructure
        self._recovery_executor = None
        self._recovery_queue: Optional[asyncio.Queue] = None   # lazily created when loop is present
        self._recovery_task: Optional[asyncio.Task] = None

        # Start recovery infra (idempotent; queue created lazily)
        self._start_recovery_system()

        self.logger.info("[OK] ErrorPinpointer initialized")

    # ───────────────────────────────────────────────────────
    # Lifecycle / Orchestrator Binding
    # ───────────────────────────────────────────────────────

    def bind_orchestrator(self, orchestrator: 'ModuleOrchestrator') -> None:
        self.orchestrator = orchestrator

    def _start_recovery_system(self) -> None:
        """Spin up thread pool & async worker for recovery tasks (idempotent)."""
        from concurrent.futures import ThreadPoolExecutor

        if self._recovery_executor is None:
            self._recovery_executor = ThreadPoolExecutor(
                max_workers=self.cfg.recovery_workers,
                thread_name_prefix="ErrorRecovery",
            )

        # Start worker if an event loop is running; queue is created lazily in _attempt_recovery
        try:
            loop = asyncio.get_running_loop()
            if self._recovery_task is None or self._recovery_task.done():
                self._recovery_task = loop.create_task(self._recovery_worker(), name="ErrorRecoveryWorker")
        except RuntimeError:
            # No loop; worker will be created once analyze_error schedules recovery
            self._recovery_task = None

        self.logger.info("[TOOL] Recovery system initialized")

    async def _recovery_worker(self) -> None:
        """Background coroutine that pulls work off _recovery_queue."""
        self.logger.info("[BOT] Recovery worker started")
        try:
            while True:
                q = self._recovery_queue
                if q is None:
                    await asyncio.sleep(0.1)
                    continue

                item = await q.get()
                try:
                    if item is None:  # shutdown token
                        q.task_done()
                        break

                    ctx: ErrorContext = item.get("context")
                    if not ctx:
                        q.task_done()
                        continue

                    self.logger.info(f"[TOOL] Processing recovery for {ctx.module_name}:{ctx.error_type}")

                    success = False
                    for act in ctx.recovery_actions:
                        try:
                            success = await self._execute_recovery_action(
                                act.get("action", ""), act.get("params", {}), ctx
                            )
                            if success:
                                self.logger.info(f"[OK] Recovery succeeded via {act.get('action')}")
                                break
                        except Exception as exc:
                            self.logger.error(f"[FAIL] Recovery action {act.get('action')} failed: {exc}")

                    key = f"{ctx.module_name}:{ctx.error_type}"
                    if success:
                        self.successful_recoveries[key] += 1
                        ctx.action_taken, ctx.action_result = True, "Automatic recovery successful"
                    else:
                        ctx.action_result = "Automatic recovery failed"

                    self.recovery_history.append(
                        {
                            "timestamp": datetime.now().isoformat(),
                            "module": ctx.module_name,
                            "error_type": ctx.error_type,
                            "success": success,
                            "actions_attempted": len(ctx.recovery_actions),
                        }
                    )
                finally:
                    try:
                        q.task_done()
                    except Exception:
                        pass
        except asyncio.CancelledError:
            self.logger.info("[STOP] Recovery worker cancelled")
        finally:
            self.logger.info("[STOP] Recovery worker exiting")

    def shutdown(self) -> None:
        """Graceful, idempotent shutdown of ErrorPinpointer infrastructure."""
        self.logger.info("[STOP] Shutting down ErrorPinpointer …")

        # Signal the worker to stop
        try:
            if self._recovery_queue is not None:
                try:
                    self._recovery_queue.put_nowait(None)
                except Exception:
                    pass
        except Exception as exc:
            self.logger.debug(f"Failed to enqueue shutdown token: {exc}")

        # Cancel task if still running
        if self._recovery_task and not self._recovery_task.done():
            try:
                self._recovery_task.cancel()
            except Exception:
                pass

        # Tear down the thread-pool
        if self._recovery_executor:
            try:
                self._recovery_executor.shutdown(wait=False)
            except Exception:
                pass

        self.logger.info("[OK] ErrorPinpointer shutdown complete")

    # ───────────────────────────────────────────────────────
    # Patterns & Recovery registration
    # ───────────────────────────────────────────────────────

    @classmethod
    def register_recovery_action(cls, name: str, handler: Callable[['ErrorPinpointer', Dict[str, Any], ErrorContext], Any]) -> None:
        cls._custom_recovery_actions[name] = handler

    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """Initialize known error patterns with recovery actions."""
        return [
            ErrorPattern(
                pattern_id="KEY_ERROR_INFOBUS",
                error_pattern=r"KeyError.*['\"]([^'\"]+)['\"]",
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
                    {'action': 'request_missing_data', 'params': {'timeout': 5.0}},
                    {'action': 'reorder_execution', 'params': {'check_dependencies': True}}
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
                    {'action': 'disable_module_temporarily', 'params': {'duration': 60}},
                    {'action': 'reduce_module_load', 'params': {'factor': 0.5}}
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
                    {'action': 'force_garbage_collection', 'params': {'generations': 2}},
                    {'action': 'clear_caches', 'params': {'preserve_critical': True}},
                    {'action': 'enter_emergency_mode', 'params': {'reason': 'Memory critical'}}
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
                    {'action': 'wait_and_reset_breaker', 'params': {'wait_time': 30}},
                    {'action': 'check_module_health', 'params': {'deep_check': True}}
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
                    {'action': 'provide_default_value', 'params': {'use_last_known': True}}
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
                    "Mutual module data requirements",
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
                    {'action': 'break_dependency_cycle', 'params': {'method': 'remove_weakest'}},
                    {'action': 'rebuild_execution_plan', 'params': {'validate': True}}
                ],
                auto_recovery=True
            ),
        ]

    # ───────────────────────────────────────────────────────
    # Error Analysis
    # ───────────────────────────────────────────────────────

    def analyze_error(self, exception: Exception, module_name: str = "Unknown") -> ErrorContext:
        """
        Comprehensive error analysis with automated recovery.
        Loop- & thread-safe; de-duplicates bursts.
        """
        start_time = time.time()

        try:
            # Basic Error Info
            error_type = type(exception).__name__
            error_message = str(exception)

            # Deduplication within window (true lightweight fast-path)
            dedupe_key = f"{module_name}:{error_type}:{error_message[:80]}"
            now = time.time()
            last = self._last_seen.get(dedupe_key, 0.0)
            deduped = (now - last) < self.cfg.dedupe_window_sec
            if deduped:
                self._last_seen[dedupe_key] = now
                context = ErrorContext(
                    error_type=error_type,
                    error_message=error_message,
                    module_name=module_name,
                    function_name="unknown",
                    file_path="unknown",
                    line_number=0,
                    timestamp=datetime.now(),
                    severity="low",
                    category="duplicate",
                )
                with self._history_lock:
                    self.error_history.append(context)
                    self.error_patterns[f"{error_type}:{module_name}"] += 1
                    self.module_error_counts[module_name] += 1
                return context

            # Traceback (use last frame where the exception occurred)
            tb = exception.__traceback__
            file_path = function_name = "unknown"
            line_number = 0
            frame = None

            if tb:
                while tb.tb_next:
                    tb = tb.tb_next
                frame = tb.tb_frame
                file_path = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = tb.tb_lineno

            # Build Context
            context = ErrorContext(
                error_type=error_type,
                error_message=error_message,
                module_name=module_name,
                function_name=function_name,
                file_path=file_path,
                line_number=line_number,
                timestamp=datetime.now(),
            )

            # Code & System Context (bounded, best-effort)
            context.source_lines = self._extract_source_lines(file_path, line_number, self.cfg.snapshot_max_source_context)
            context.local_variables = self._extract_local_variables(frame, self.cfg.snapshot_max_locals)
            context.call_stack = self._extract_call_stack(exception.__traceback__)
            context.module_state = self._get_module_state(module_name)
            context.infobus_snapshot = self._get_infobus_snapshot()
            context.related_errors = self._find_related_errors(error_type, module_name)

            # Pattern analysis + fixes + actions
            self._analyze_error_pattern(context)
            context.recovery_actions = self._generate_recovery_actions(context)
            context.suggested_fixes = self._generate_fix_suggestions(context)
            context.reproduction_steps = self._generate_reproduction_steps(context)

            # Record & correlate (locked)
            with self._history_lock:
                self.error_history.append(context)
                self.error_patterns[f"{error_type}:{module_name}"] += 1
                self.module_error_counts[module_name] += 1

            self._correlate_error(context)

            # Auto-recovery scheduling (loop-aware)
            if context.recovery_actions and self._should_attempt_recovery(context):
                try:
                    loop = asyncio.get_running_loop()
                    # Ensure queue bound to *this* loop and worker running
                    if self._recovery_queue is None:
                        self._recovery_queue = asyncio.Queue(maxsize=self.cfg.recovery_queue_size)
                    if self._recovery_task is None or self._recovery_task.done():
                        self._recovery_task = loop.create_task(self._recovery_worker(), name="ErrorRecoveryWorker")
                    loop.create_task(self._attempt_recovery(context))
                except RuntimeError:
                    # No loop; fall back to sync recovery in thread pool
                    if self._recovery_executor:
                        self._recovery_executor.submit(self._sync_recovery, context)

            # Perf
            analysis_time = (time.time() - start_time) * 1000
            with self._history_lock:
                self.analysis_times.append(analysis_time)

            # Log (rate-limited effect handled above)
            self.logger.error(format_operator_message(
                "[CRASH]",
                message=f"ERROR ANALYZED: {error_type} in {module_name}::{function_name}:{line_number}",
                severity=context.severity,
                category=context.category,
                recovery_planned=len(context.recovery_actions) > 0
            ))

            # Mark dedupe timestamp
            self._last_seen[dedupe_key] = now
            return context

        except Exception as analysis_error:
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
                suggested_fixes=["Manual debugging required - error analysis failed"],
            )

    # ───────────────────────────────────────────────────────
    # Recovery (attempt & execution)
    # ───────────────────────────────────────────────────────

    def _should_attempt_recovery(self, context: ErrorContext) -> bool:
        """Determine if automatic recovery should be attempted."""
        recovery_key = f"{context.module_name}:{context.error_type}"
        attempts = self.recovery_attempts[recovery_key]

        if attempts > self.cfg.max_recovery_attempts_per_key:
            ratio = self.successful_recoveries[recovery_key] / max(attempts, 1)
            if ratio < self.cfg.min_success_ratio_for_retries:
                return False

        # Pattern flag
        msg = context.error_message
        for pattern in self.known_patterns:
            rx = self._compiled_regex.get(pattern.pattern_id)
            if rx and rx.search(msg):
                return pattern.auto_recovery
        return False

    async def _attempt_recovery(self, context: ErrorContext) -> None:
        """Queue recovery attempt for background processing."""
        recovery_key = f"{context.module_name}:{context.error_type}"
        self.recovery_attempts[recovery_key] += 1

        # Ensure infra bound to the current loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self.logger.warning("[WARN] No running loop; skipping async recovery enqueue")
            return

        if self._recovery_queue is None:
            self._recovery_queue = asyncio.Queue(maxsize=self.cfg.recovery_queue_size)
        if self._recovery_task is None or self._recovery_task.done():
            self._recovery_task = loop.create_task(self._recovery_worker(), name="ErrorRecoveryWorker")

        # Enqueue (bounded)
        q = self._recovery_queue
        try:
            await asyncio.wait_for(q.put({'context': context, 'timestamp': time.time()}), timeout=1.0)
            self.logger.info(f"📋 Queued recovery for {context.module_name}")
        except asyncio.TimeoutError:
            self.logger.warning(f"[WARN] Recovery queue full for {context.module_name}")

    def _sync_recovery(self, context: ErrorContext) -> None:
        """Synchronous recovery fallback when no event loop available."""
        try:
            self.logger.info(f"[TOOL] Sync recovery for {context.module_name}")

            for action in context.recovery_actions:
                if self._execute_recovery_action_sync(action.get('action', ''), action.get('params', {}), context):
                    self.successful_recoveries[f"{context.module_name}:{context.error_type}"] += 1
                    context.action_taken, context.action_result = True, "Automatic recovery successful"
                    return
            context.action_result = "Automatic recovery failed"
        except Exception as e:
            self.logger.error(f"Sync recovery failed: {e}")

    def _execute_recovery_action_sync(self, action_type: str, params: Dict[str, Any], context: ErrorContext) -> bool:
        """Sync execution for environments without a running loop (best-effort subset)."""
        # Try custom handler first
        handler = self._custom_recovery_actions.get(action_type)
        if handler:
            try:
                return bool(handler(self, params, context))
            except Exception as e:
                self.logger.error(f"Custom recovery '{action_type}' failed: {e}")
                return False

        # Built-ins (subset)
        try:
            if action_type == 'force_garbage_collection':
                import gc
                generations = int(params.get('generations', 2))
                for _ in range(max(generations, 1)):
                    gc.collect()
                # Best-effort SmartInfoBus cleanup
                bus = InfoBusManager.get_instance()
                cleanup = getattr(bus, "_cleanup_old_data", None)
                if callable(cleanup):
                    try:
                        cleanup()
                    except Exception:
                        pass
                return True

            if action_type == 'enter_emergency_mode':
                orc = self.orchestrator
                if orc and hasattr(orc, 'trigger_emergency_mode_manually'):
                    getattr(orc, 'trigger_emergency_mode_manually')(params.get('reason', f"Error in {context.module_name}"))
                    return True
        except Exception as e:
            self.logger.error(f"Sync recovery action {action_type} failed: {e}")
        return False

    async def _execute_recovery_action(self, action_type: str, params: Dict[str, Any], context: ErrorContext) -> bool:
        """Execute specific recovery action (async path)."""
        # Custom action hook
        handler = self._custom_recovery_actions.get(action_type)
        if handler:
            try:
                res = handler(self, params, context)
                if inspect.isawaitable(res):
                    res = await res
                return bool(res)
            except Exception as e:
                self.logger.error(f"Custom recovery '{action_type}' failed: {e}")
                return False

        orc = self.orchestrator
        try:
            if action_type == 'request_missing_data':
                key_match = re.search(r"['\"]([^'\"]+)['\"]", context.error_message)
                if key_match:
                    key = key_match.group(1)
                    bus = InfoBusManager.get_instance()
                    requester = getattr(bus, "request_data", None)
                    if callable(requester):
                        requester(key, context.module_name)
                    await asyncio.sleep(float(params.get('timeout', 5.0)))
                    getter = getattr(bus, "get", None)
                    if callable(getter) and getter(key) is not None:  # type: ignore[arg-type]
                        return True
                    store = getattr(bus, "_data_store", None)
                    if isinstance(store, dict) and key in store:
                        return True
                return False

            if action_type == 'disable_module_temporarily' and orc:
                duration = int(params.get('duration', 60))
                if hasattr(orc, 'disable_module'):
                    orc.disable_module(context.module_name)

                async def re_enable():
                    await asyncio.sleep(duration)
                    if hasattr(orc, 'enable_module'):
                        orc.enable_module(context.module_name)

                asyncio.create_task(re_enable())
                return True

            if action_type == 'force_garbage_collection':
                import gc
                generations = int(params.get('generations', 2))
                for _ in range(max(generations, 1)):
                    gc.collect()
                if params.get('preserve_critical', True):
                    bus = InfoBusManager.get_instance()
                    cleanup = getattr(bus, "_cleanup_old_data", None)
                    if callable(cleanup):
                        try:
                            cleanup()
                        except Exception:
                            pass
                return True

            if action_type == 'enter_emergency_mode' and orc:
                reason = params.get('reason', f"Error in {context.module_name}")
                trigger = getattr(orc, 'trigger_emergency_mode_manually', None)
                if callable(trigger):
                    trigger(reason)
                return True

            if action_type == 'wait_and_reset_breaker' and orc:
                wait_time = int(params.get('wait_time', 30))
                await asyncio.sleep(wait_time)
                reset = getattr(orc, 'reset_circuit_breaker', None)
                return bool(callable(reset) and reset(context.module_name))

            if action_type == 'break_dependency_cycle' and orc:
                rebuild = getattr(orc, 'build_execution_plan', None)
                if callable(rebuild):
                    rebuild()
                return True

            if action_type == 'rebuild_execution_plan' and orc:
                # Always rebuild first if available
                rebuild = getattr(orc, 'build_execution_plan', None)
                if callable(rebuild):
                    rebuild()

                # Pylance-safe validation call (private or public name)
                if bool(params.get('validate', True)):
                    validator = (
                        getattr(orc, '_validate_system_integrity', None) or
                        getattr(orc, 'validate_system_integrity', None)
                    )
                    if callable(validator):
                        validator()
                return True

            if action_type == 'reduce_module_load' and orc:
                factor = float(params.get('factor', 0.5))
                module = getattr(orc, 'modules', {}).get(context.module_name) if hasattr(orc, 'modules') else None
                if module and hasattr(module, 'reduce_load'):
                    module.reduce_load(factor)
                    return True

            if action_type == 'clear_caches':
                bus = InfoBusManager.get_instance()
                clear_fn = getattr(bus, "clear_caches", None)
                if callable(clear_fn):
                    clear_fn(preserve_critical=bool(params.get('preserve_critical', True)))
                    return True
                return True  # nothing to clear is fine

            if action_type == 'provide_default_value':
                # Guidance-only; nothing to execute globally
                return False

            if action_type == 'check_module_health' and orc:
                module = getattr(orc, 'modules', {}).get(context.module_name) if hasattr(orc, 'modules') else None
                if module and hasattr(module, 'get_health_status'):
                    health = module.get_health_status()
                    return health.get('status') != 'CRITICAL'
                return True

            if action_type == 'reorder_execution' and orc:
                build = getattr(orc, 'build_execution_plan', None)
                if callable(build):
                    build()
                return True

        except Exception as e:
            self.logger.error(f"Recovery action {action_type} failed: {e}")

        return False

    # ───────────────────────────────────────────────────────
    # Pattern Analysis & Suggestions
    # ───────────────────────────────────────────────────────

    def _analyze_error_pattern(self, context: ErrorContext) -> None:
        """Analyze error against known patterns with caching."""
        cache_key = f"{context.error_type}:{context.error_message[:80]}"

        with self._cache_lock:
            cached = self._pattern_cache.get(cache_key)
            if cached:
                context.category = cached['category']
                context.severity = cached['severity']
                context.suggested_fixes.extend(cached['fixes'])
                return

        for pattern in self.known_patterns:
            rx = self._compiled_regex.get(pattern.pattern_id)
            if rx and rx.search(context.error_message):
                context.category = pattern.category
                context.severity = pattern.severity
                context.suggested_fixes.extend(pattern.fix_suggestions)
                with self._cache_lock:
                    self._pattern_cache[cache_key] = {
                        'category': pattern.category,
                        'severity': pattern.severity,
                        'fixes': pattern.fix_suggestions
                    }
                return

        # Fallback classification
        msg = context.error_message.lower()
        if "timeout" in msg:
            context.category, context.severity = "timeout", "high"
        elif "memory" in msg:
            context.category, context.severity = "resource", "critical"
        elif context.error_type in ("KeyError", "AttributeError"):
            context.category, context.severity = "data", "medium"
        else:
            context.category, context.severity = "logic", "medium"

    def _generate_recovery_actions(self, context: ErrorContext) -> List[Dict[str, Any]]:
        """Generate recovery actions based on error pattern and severity."""
        actions: List[Dict[str, Any]] = []
        for pattern in self.known_patterns:
            rx = self._compiled_regex.get(pattern.pattern_id)
            if rx and rx.search(context.error_message):
                actions.extend(pattern.recovery_actions)
                break

        if not actions:
            if context.severity == "critical":
                actions.append({'action': 'enter_emergency_mode', 'params': {'reason': f"Critical error in {context.module_name}"}})
            elif context.severity == "high":
                actions.append({'action': 'disable_module_temporarily', 'params': {'duration': 30}})
        return actions

    def _generate_fix_suggestions(self, context: ErrorContext) -> List[str]:
        """Generate specific fix suggestions based on error context."""
        if context.suggested_fixes:
            return context.suggested_fixes

        suggestions: List[str] = [f"1. Check line {context.line_number} in {Path(context.file_path).name}"]

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
                "5. Use generators for large data"
            ])

        if context.recovery_actions:
            suggestions.append(f"{len(suggestions)+1}. Automatic recovery available - monitor results")
        return suggestions

    def _generate_reproduction_steps(self, context: ErrorContext) -> List[str]:
        steps = [
            f"1. Load module: {context.module_name}",
            f"2. Execute function: {context.function_name}",
            f"3. Monitor line: {context.line_number}",
        ]
        if context.local_variables:
            steps.append("4. Set local variables:")
            for var, value in list(context.local_variables.items())[:3]:
                steps.append(f"   {var} = {value}")
        if "KeyError" in context.error_type:
            steps.append("5. Check InfoBus state for missing keys")
        if context.module_state.get('circuit_breaker'):
            cb = context.module_state['circuit_breaker']
            steps.append(f"6. Circuit breaker state: {cb.get('state')}")
        return steps

    # ───────────────────────────────────────────────────────
    # Correlation & Snapshots
    # ───────────────────────────────────────────────────────

    def _correlate_error(self, context: ErrorContext) -> None:
        """Correlate error with recent system events."""
        correlation_key = f"{context.error_type}:{context.module_name}"
        recent_errors = list(self.error_history)[-20:]
        correlated = []

        for err in recent_errors:
            if err is context:
                continue
            time_diff = abs((context.timestamp - err.timestamp).total_seconds())
            if time_diff < 5:
                correlated.append({'error': f"{err.error_type} in {err.module_name}", 'time_diff': time_diff, 'severity': err.severity})

        if correlated:
            self.error_correlations[correlation_key].extend(correlated)
            context.related_errors.extend([c['error'] for c in correlated[:3]])

    def _extract_source_lines(self, file_path: str, line_number: int, context_lines: int = 10) -> List[str]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            out: List[str] = []
            for i in range(start, end):
                marker = ">>> " if i == line_number - 1 else "    "
                out.append(f"{marker}{i+1:4d}: {lines[i].rstrip()}")
            return out
        except Exception as e:
            return [f"Could not read source file: {e}"]

    def _extract_local_variables(self, frame, max_items: int = 50) -> Dict[str, Any]:
        """Extract local variables with safe serialization."""
        local_vars: Dict[str, Any] = {}
        if not frame:
            return local_vars

        try:
            for idx, (var_name, var_value) in enumerate(frame.f_locals.items()):
                if idx >= max_items:
                    break
                if var_name.startswith('__'):
                    continue
                try:
                    if isinstance(var_value, (int, float, bool, type(None))):
                        local_vars[var_name] = var_value
                    elif isinstance(var_value, (str, bytes)):
                        if len(var_value) > 1000:
                            local_vars[var_name] = f"{type(var_value).__name__}(length={len(var_value)})"
                        else:
                            local_vars[var_name] = var_value
                    elif isinstance(var_value, (list, dict, set, tuple)):
                        local_vars[var_name] = f"{type(var_value).__name__}(length={len(var_value)})"
                    elif NUMPY_AVAILABLE and np is not None and isinstance(var_value, np.ndarray):
                        local_vars[var_name] = f"ndarray(shape={var_value.shape}, dtype={var_value.dtype})"
                    else:
                        local_vars[var_name] = f"{type(var_value).__name__}"
                except Exception:
                    local_vars[var_name] = "<Unable to serialize>"
        except Exception:
            pass
        return local_vars

    def _extract_call_stack(self, tb) -> List[Dict[str, Any]]:
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
        """Get current module state if available (duck-typed)."""
        orc = self.orchestrator
        if not orc:
            return {}

        try:
            module = None
            if hasattr(orc, 'get_module_by_name'):
                module = orc.get_module_by_name(module_name)
            elif hasattr(orc, 'modules'):
                module = getattr(orc, 'modules', {}).get(module_name)  # type: ignore[attr-defined]

            if not module:
                return {}

            state: Dict[str, Any] = {}
            if hasattr(module, 'get_state'):
                try:
                    state = module.get_state() or {}
                except Exception:
                    state = {}

            if hasattr(orc, 'get_circuit_breaker_status'):
                try:
                    cb_status = orc.get_circuit_breaker_status()
                    if isinstance(cb_status, dict) and module_name in cb_status:
                        state['circuit_breaker'] = cb_status[module_name]
                except Exception:
                    pass

            if hasattr(orc, 'module_performance'):
                try:
                    perf = getattr(orc, 'module_performance', {})
                    if isinstance(perf, dict) and module_name in perf:
                        state['performance'] = perf[module_name]
                except Exception:
                    pass

            return state
        except Exception:
            return {"error": "Failed to get module state"}

    def _get_infobus_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current InfoBus state (best-effort, guarded)."""
        try:
            bus = InfoBusManager.get_instance()
            data_store = getattr(bus, "_data_store", {})
            module_disabled = getattr(bus, "_module_disabled", set())
            event_log = getattr(bus, "_event_log", deque(maxlen=0))
            get_perf = getattr(bus, "get_performance_metrics", None)

            snapshot: Dict[str, Any] = {
                'data_keys': list(data_store.keys())[:20] if isinstance(data_store, dict) else [],
                'total_keys': len(data_store) if isinstance(data_store, dict) else 0,
                'disabled_modules': list(module_disabled) if isinstance(module_disabled, (set, list)) else [],
                'recent_events': list(event_log)[-10:] if isinstance(event_log, (list, deque)) else [],
                'performance_metrics': get_perf() if callable(get_perf) else {},
            }

            if PSUTIL_AVAILABLE and psutil is not None:
                try:
                    snapshot['memory_usage_mb'] = round(psutil.Process().memory_info().rss / 1024 / 1024, 2)
                    snapshot['cpu_percent'] = psutil.cpu_percent(interval=0.05)
                except Exception:
                    pass

            return snapshot
        except Exception as e:
            return {"error": f"Failed to get InfoBus snapshot: {e}"}

    def _find_related_errors(self, error_type: str, module_name: str, lookback_minutes: int = 10) -> List[str]:
        related: List[str] = []
        cutoff = datetime.now().timestamp() - (lookback_minutes * 60)
        for err in self.error_history:
            if err.timestamp.timestamp() < cutoff:
                continue
            if err.error_type == error_type and err.module_name != module_name:
                related.append(f"Same error in {err.module_name}")
            elif err.module_name == module_name and err.error_type != error_type:
                related.append(f"{err.error_type} in same module")
            if err.severity == "critical" and abs((err.timestamp - datetime.now()).total_seconds()) < 30:
                related.append(f"Critical error: {err.error_type} in {err.module_name}")
        return related[:5]

    # ───────────────────────────────────────────────────────
    # Reporting & Guides
    # ───────────────────────────────────────────────────────

    def create_debugging_guide(self, context: ErrorContext) -> str:
        guide = f"""
[SEARCH] DEBUGGING GUIDE: {context.error_type}
{'=' * 60}

📍 ERROR LOCATION:
  Module: {context.module_name}
  Function: {context.function_name}
  File: {Path(context.file_path).name}
  Line: {context.line_number}

[WARN] ERROR DETAILS:
  Type: {context.error_type}
  Message: {context.error_message}
  Severity: {context.severity.upper()}
  Category: {context.category}

[STATS] SYSTEM STATE:
  Timestamp: {context.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
  Module Errors: {self.module_error_counts.get(context.module_name, 0)}
  Related Errors: {len(context.related_errors)}

💡 SUGGESTED FIXES:
"""
        for i, fix in enumerate(context.suggested_fixes, 1):
            guide += f"  {i}. {fix}\n"

        guide += "\n[RELOAD] REPRODUCTION STEPS:\n"
        for i, step in enumerate(context.reproduction_steps, 1):
            guide += f"  {i}. {step}\n"

        if context.recovery_actions:
            guide += "\n[TOOL] AUTOMATED RECOVERY:\n"
            for action in context.recovery_actions:
                guide += f"  • {action.get('action')} "
                params = action.get('params') or {}
                if params:
                    guide += f"({', '.join(f'{k}={v}' for k, v in params.items())})"
                guide += "\n"

        if context.source_lines:
            guide += "\n[LOG] CODE CONTEXT:\n```python\n"
            guide += "\n".join(context.source_lines[:10])
            guide += "\n```\n"

        if context.local_variables:
            guide += "\n🔢 LOCAL VARIABLES:\n"
            for var, value in list(context.local_variables.items())[:5]:
                guide += f"  {var} = {value}\n"

        if self.orchestrator:
            try:
                open_breakers = 0
                cbs = getattr(self.orchestrator, 'circuit_breakers', {})
                if isinstance(cbs, dict):
                    open_breakers = sum(1 for cb in cbs.values() if getattr(cb, 'state', 'CLOSED') == 'OPEN')
                emergency = False
                if hasattr(self.orchestrator, 'get_emergency_mode_status'):
                    status = self.orchestrator.get_emergency_mode_status()
                    emergency = bool(status.get('active')) if isinstance(status, dict) else bool(getattr(status, 'active', False))
                guide += f"""
🚦 ORCHESTRATOR STATUS
{'─' * 30}
- Emergency Mode: {'ACTIVE' if emergency else 'Inactive'}
- Circuit Breakers Open: {open_breakers}
- Total Modules: {len(getattr(self.orchestrator, 'modules', {}))}
"""
            except Exception:
                pass

        guide += """

💻 QUICK ACTIONS:
  1. Check module logs: error_pinpointer.get_module_error_log(module_name)
  2. View error history: error_pinpointer.get_error_summary()
  3. Check correlations: error_pinpointer.correlate_errors()
"""
        if self.orchestrator:
            guide += "  4. Reset module: orchestrator.reset_circuit_breaker(module_name)\n"
            guide += "  5. Disable module: orchestrator.disable_module(module_name)\n"
        return guide

    def get_error_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            'total_errors': len(self.error_history),
            'unique_modules': len(set(e.module_name for e in self.error_history)),
            'most_common_errors': [],
            'problem_modules': [],
            'error_trends': {},
            'avg_analysis_time_ms': _mean(list(self.analysis_times)),
            'recovery_stats': {
                'total_attempts': sum(self.recovery_attempts.values()),
                'successful': sum(self.successful_recoveries.values()),
                'success_rate': (sum(self.successful_recoveries.values()) / max(sum(self.recovery_attempts.values()), 1))
            }
        }

        error_counts: Dict[str, int] = {}
        module_errors: Dict[str, int] = {}
        severity_counts: Dict[str, int] = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'duplicate': 0}

        for error in self.error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            module_errors[error.module_name] = module_errors.get(error.module_name, 0) + 1
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1

        summary['most_common_errors'] = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        summary['problem_modules'] = sorted(module_errors.items(), key=lambda x: x[1], reverse=True)[:5]
        summary['severity_distribution'] = severity_counts

        hour_ago = datetime.now().timestamp() - 3600
        recent_errors = [e for e in self.error_history if e.timestamp.timestamp() > hour_ago]
        if recent_errors:
            intervals = defaultdict(int)
            for error in recent_errors:
                interval = int(error.timestamp.timestamp() / 300) * 300
                intervals[interval] += 1
            summary['error_trends'] = dict(sorted(intervals.items()))
        return summary

    def get_module_error_log(self, module_name: str) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for error in self.error_history:
            if error.module_name == module_name:
                msg = error.error_message
                out.append({
                    'timestamp': error.timestamp.isoformat(),
                    'error_type': error.error_type,
                    'message': (msg[:100] + '...') if len(msg) > 100 else msg,
                    'severity': error.severity,
                    'category': error.category,
                    'line': error.line_number,
                    'function': error.function_name,
                    'recovery_attempted': error.action_taken,
                    'recovery_result': error.action_result
                })
        return out

    def correlate_errors(self, timeframe_minutes: int = 10) -> List[ErrorPattern]:
        """Cluster recent errors and return dynamic ErrorPattern objects."""
        patterns: List[ErrorPattern] = []
        cutoff = datetime.now().timestamp() - timeframe_minutes * 60
        groups: Dict[str, List[ErrorContext]] = defaultdict(list)

        for err in self.error_history:
            if err.timestamp.timestamp() >= cutoff:
                groups[f"{err.error_type}:{err.module_name}"].append(err)

        for key, errs in groups.items():
            if len(errs) < 2:
                continue

            span = max((errs[-1].timestamp - errs[0].timestamp).total_seconds(), 1.0)
            freq = len(errs) / (span / 60.0)

            dep_sets = [set(e.module_state.get("dependencies", [])) for e in errs]
            common_deps = set.intersection(*dep_sets) if dep_sets and all(dep_sets) else set()

            patterns.append(
                ErrorPattern(
                    pattern_id=f"dynamic_{key}_{int(time.time())}",
                    error_pattern=key,
                    category=errs[0].category,
                    severity="critical" if freq > 1 else "high",
                    description=f"Recurring {errs[0].error_type} in {errs[0].module_name}",
                    common_causes=[f"Frequency ≈ {freq:.2f}/min", f"Common deps: {', '.join(common_deps) or 'None'}"],
                    fix_suggestions=[f"Profile {errs[0].module_name}", "Inspect shared dependencies", "Review recent commits"],
                    prevention_tips=["Improve input validation", "Add back-pressure / throttling"],
                    recovery_actions=[{"action": "disable_module_temporarily", "params": {"duration": 300}}],
                    auto_recovery=True,
                )
            )

        # Push into SmartInfoBus (best-effort)
        if patterns:
            try:
                bus = InfoBusManager.get_instance()
                setter = getattr(bus, "set", None)
                if callable(setter):
                    setter(
                        "error_patterns",
                        {
                            "timestamp": datetime.now().isoformat(),
                            "patterns": [p.__dict__ for p in patterns],
                            "action_required": len(patterns) > 5,
                        },
                        module="ErrorPinpointer",
                        thesis=f"Detected {len(patterns)} error patterns",
                    )
            except Exception:
                pass

        return patterns

    def create_debug_snapshot(self, module_name: str) -> Dict[str, Any]:
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'recent_errors': self.get_module_error_log(module_name)[-10:],
            'module_state': self._get_module_state(module_name),
            'infobus_state': self._get_infobus_snapshot(),
            'execution_context': {},
            'recovery_history': []
        }

        if self.orchestrator:
            try:
                cb_obj = getattr(self.orchestrator, 'circuit_breakers', {}).get(module_name)
                cb_stats = cb_obj.get_stats() if cb_obj and hasattr(cb_obj, "get_stats") else {}
                snapshot['execution_context'] = {
                    'is_enabled': module_name not in getattr(InfoBusManager.get_instance(), "_module_disabled", set()),
                    'circuit_breaker': cb_stats,
                }
                if hasattr(self.orchestrator, 'module_performance') and module_name in self.orchestrator.module_performance:
                    snapshot['performance'] = self.orchestrator.module_performance[module_name]
            except Exception:
                pass

        for record in self.recovery_history:
            if record.get('module') == module_name:
                snapshot['recovery_history'].append(record)

        return snapshot

    def export_error_report(self, filepath: str, last_n_errors: int = 100) -> None:
        """Export comprehensive error report to JSON (directories auto-created)."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_errors_tracked': len(self.error_history),
                'analysis_period': f'Last {last_n_errors} errors',
                'avg_analysis_time_ms': _mean(list(self.analysis_times)),
                'system_info': {}
            },
            'summary': self.get_error_summary(),
            'detailed_errors': [error.to_dict() for error in list(self.error_history)[-last_n_errors:]],
            'patterns': dict(self.error_patterns),
            'module_statistics': dict(self.module_error_counts),
            'correlations': dict(self.error_correlations),
            'recovery_statistics': {
                'attempts': dict(self.recovery_attempts),
                'successes': dict(self.successful_recoveries),
                'recent_recoveries': list(self.recovery_history)[-20:]
            }
        }

        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                report['metadata']['system_info'] = {
                    'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                    'cpu_percent': psutil.cpu_percent(interval=0.05)
                }
            except Exception:
                pass

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"[STATS] Error report exported to {filepath}")

    def get_debugging_guide(self, error_type: Optional[str] = None, module_name: Optional[str] = None) -> str:
        guide = f"""
[SEARCH] SMARTINFOBUS DEBUGGING GUIDE
{'=' * 50}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target: {error_type or 'All'} errors in {module_name or 'All'} modules

📋 QUICK DEBUGGING CHECKLIST
{'─' * 30}
1. [OK] Check module execution order
2. [OK] Verify InfoBus data availability
3. [OK] Validate input parameters
4. [OK] Review error context and local variables
5. [OK] Check for related errors in other modules
6. [OK] Examine system resource usage
7. [OK] Verify circuit breaker states
8. [OK] Check for emergency mode activation

🛠️ COMMON ERROR PATTERNS
{'─' * 30}
"""
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

        if module_name and module_name in self.module_error_counts:
            error_count = self.module_error_counts[module_name]
            guide += f"""

[TARGET] MODULE-SPECIFIC INSIGHTS: {module_name}
{'─' * 30}
- Total errors: {error_count}
- Most common: {self._get_most_common_error_for_module(module_name)}
- Status: {'[WARN] High error rate' if error_count > 10 else '[OK] Normal'}
- Recovery success rate: {self._get_recovery_rate_for_module(module_name):.1%}
"""

        if self.orchestrator:
            try:
                open_breakers = 0
                cbs = getattr(self.orchestrator, 'circuit_breakers', {})
                if isinstance(cbs, dict):
                    open_breakers = sum(1 for cb in cbs.values() if getattr(cb, 'state', 'CLOSED') == 'OPEN')
                emergency = False
                if hasattr(self.orchestrator, 'get_emergency_mode_status'):
                    status = self.orchestrator.get_emergency_mode_status()
                    emergency = bool(status.get('active')) if isinstance(status, dict) else bool(getattr(status, 'active', False))
                guide += f"""

🚦 SYSTEM STATUS
{'─' * 30}
- Emergency Mode: {'ACTIVE' if emergency else 'Inactive'}
- Circuit Breakers Open: {open_breakers}
- Total Modules: {len(getattr(self.orchestrator, 'modules', {}))}
"""
            except Exception:
                pass

        guide += """

[ROCKET] ADVANCED DEBUGGING TOOLS
─────────────────────────────
1. error_pinpointer.create_debug_snapshot(module_name)
2. error_pinpointer.correlate_errors(timeframe_minutes)
3. error_pinpointer.export_error_report(filepath)
4. orchestrator.get_system_status_report()
5. smart_bus.get_performance_metrics()

[TOOL] RECOVERY TOOLS
─────────────────
1. orchestrator.reset_circuit_breaker(module_name)
2. orchestrator.exit_emergency_mode()
3. orchestrator.enable_module(module_name)
4. smart_bus.cleanup_old_data()
"""
        return guide

    def _get_most_common_error_for_module(self, module_name: str) -> str:
        module_errors = defaultdict(int)
        for error in self.error_history:
            if error.module_name == module_name:
                module_errors[error.error_type] += 1
        return max(module_errors.items(), key=lambda x: x[1])[0] if module_errors else "No errors recorded"

    def _get_recovery_rate_for_module(self, module_name: str) -> float:
        attempts = successes = 0
        for key, count in self.recovery_attempts.items():
            if key.startswith(f"{module_name}:"):
                attempts += count
                successes += self.successful_recoveries.get(key, 0)
        return successes / max(attempts, 1)

    def clear_history(self, keep_last_n: int = 100) -> None:
        """Clear error history, keeping only recent errors."""
        if len(self.error_history) > keep_last_n:
            recent_errors = list(self.error_history)[-keep_last_n:]
            self.error_history = deque(recent_errors, maxlen=self.cfg.max_history)
            with self._cache_lock:
                self._pattern_cache.clear()
            self.logger.info(f"🧹 Cleared error history, kept last {keep_last_n} errors")


# ═══════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def analyze_exception(exception: Exception, module_name: str = "Unknown") -> ErrorContext:
    pinpointer = ErrorPinpointer()
    return pinpointer.analyze_error(exception, module_name)


def create_error_handler(module_name: str, pinpointer: Optional[ErrorPinpointer] = None):
    """Create error handler decorator for modules (sync + async) with metadata preserved."""
    pin = pinpointer or ErrorPinpointer()

    def error_handler(func: Callable):
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    ctx = pin.analyze_error(e, module_name)
                    print(pin.create_debugging_guide(ctx))
                    raise

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                ctx = pin.analyze_error(e, module_name)
                print(pin.create_debugging_guide(ctx))
                raise

        return sync_wrapper

    return error_handler
