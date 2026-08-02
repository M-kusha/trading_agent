# modules/reward/debug/reward_debug_manager.py
"""
Comprehensive Debug Manager for Reward System
Provides detailed debugging, bus inspection, and clear English reporting
"""

from __future__ import annotations

import threading
import time
import traceback
import tracemalloc
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


class RewardDebugManager:
    """
    Comprehensive debug manager for reward system

    Features:
    - Clear English logging of all operations
    - SmartInfoBus key validation and tracking
    - Missing/broken data detection
    - Performance & memory profiling (optional)
    - Detailed error tracking
    - State inspection
    - Thread-safe updates
    """

    # ----------------------------
    # Construction & configuration
    # ----------------------------

    def __init__(
        self,
        enabled: bool = False,
        level: str = "TRACE",
        logger: Any = None,
        smart_bus: Any = None
    ):
        """Initialize debug manager"""

        self.enabled = enabled
        self.level = level
        self.logger = logger
        self.smart_bus = smart_bus

        # Debug levels (lower = more verbose)
        self.levels: Dict[str, int] = {
            'TRACE': 0,    # Everything
            'DEBUG': 1,    # Detailed
            'INFO': 2,     # Important
            'WARNING': 3,  # Issues
            'ERROR': 4     # Errors only
        }
        self.current_level: int = self.levels.get(level.upper(), 2)

        # Thread safety
        self._lock = threading.RLock()

        # Optional memory profiling
        self._mem_enabled = False

        # Bus key tracking
        self.expected_keys: Dict[str, Dict[str, Any]] = {
            # Core trading data
            'trade_data': {'source': 'TradingSystem', 'required': True},
            'trades': {'source': 'PositionManager', 'required': False},
            'recent_trades': {'source': 'TradingSystem', 'required': False},

            # Risk metrics
            'risk_metrics': {'source': 'RiskManager', 'required': True},
            'account_state': {'source': 'AccountManager', 'required': False},

            # Market context
            'market_context': {'source': 'MarketAnalyzer', 'required': True},
            'market_state': {'source': 'MarketStateManager', 'required': False},
            'market_regime': {'source': 'RegimeDetector', 'required': False},
            'regime_prediction': {'source': 'RegimePredictor', 'required': False},

            # Performance
            'performance_data': {'source': 'PerformanceTracker', 'required': False},
            'environment_config': {'source': 'Environment', 'required': False},

            # Mistake memory
            'mistake_memory': {'source': 'MistakeMemory', 'required': False},
        }

        # Tracking
        self.missing_keys_history: deque = deque(maxlen=100)
        self.broken_data_history: deque = deque(maxlen=100)
        self.key_access_counts: Dict[str, int] = defaultdict(int)
        self.key_miss_counts: Dict[str, int] = defaultdict(int)
        self.key_error_counts: Dict[str, int] = defaultdict(int)

        # Performance tracking
        # Rolling history per operation name (FIX: use deque with maxlen to prevent memory leak)
        self.operation_timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        # Per-process breakdown (overwritten each process)
        self.calculation_breakdown: Dict[str, float] = {}

        # Error tracking
        self.error_history: deque = deque(maxlen=50)
        self.error_patterns: Dict[str, int] = defaultdict(int)

        # State snapshots
        self.state_snapshots: deque = deque(maxlen=20)
        self.reward_history: deque = deque(maxlen=100)

        # Statistics
        self.total_processes: int = 0
        self.successful_processes: int = 0
        self.failed_processes: int = 0
        self.fallback_processes: int = 0

        # Internal helpers
        self._last_log_time: Dict[Tuple[str, str], float] = {}  # (level, context) -> ts
        self._current_process_id: Optional[str] = None

    # ----------------------------
    # Runtime controls
    # ----------------------------

    def enable(self) -> None:
        with self._lock:
            self.enabled = True

    def disable(self) -> None:
        with self._lock:
            self.enabled = False

    def set_level(self, level: str) -> None:
        with self._lock:
            self.level = level.upper()
            self.current_level = self.levels.get(self.level, 2)

    def enable_memory_tracking(self) -> None:
        """Enable tracemalloc for memory profiling."""
        with self._lock:
            if not tracemalloc.is_tracing():
                tracemalloc.start()
            self._mem_enabled = True

    def disable_memory_tracking(self) -> None:
        with self._lock:
            self._mem_enabled = False

    # ----------------------------
    # Timers & Profilers
    # ----------------------------

    @contextmanager
    def time_block(self, operation: str):
        """
        Context manager to time a block and record timings.
        Also updates the per-process breakdown.
        """
        start_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            end_ns = time.perf_counter_ns()
            ms = (end_ns - start_ns) / 1_000_000.0
            with self._lock:
                self.operation_timings[operation].append(ms)
                self.calculation_breakdown[operation] = self.calculation_breakdown.get(operation, 0.0) + ms

    def profiled(self, operation: str) -> Callable:
        """
        Decorator to profile a function and record its timing.
        Usage:
            @debug_manager.profiled("my_operation")
            def work(...): ...
        """
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                start_ns = time.perf_counter_ns()
                try:
                    return func(*args, **kwargs)
                finally:
                    end_ns = time.perf_counter_ns()
                    ms = (end_ns - start_ns) / 1_000_000.0
                    with self._lock:
                        self.operation_timings[operation].append(ms)
                        self.calculation_breakdown[operation] = self.calculation_breakdown.get(operation, 0.0) + ms
            return wrapper
        return decorator

    @contextmanager
    def memory_block(self, operation: str):
        """
        Context manager to sample memory deltas via tracemalloc (if enabled).
        Adds a pseudo timing entry named f"{operation}#mem_kb" with delta in KB.
        """
        if not self._mem_enabled or not tracemalloc.is_tracing():
            # No-op if memory tracking disabled
            yield
            return
        snapshot_before = tracemalloc.take_snapshot()
        try:
            yield
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            delta_bytes = sum([s.size_diff for s in stats])
            delta_kb = delta_bytes / 1024.0
            with self._lock:
                key = f"{operation}#mem_kb"
                self.operation_timings[key].append(delta_kb)
                # Do not add to calculation_breakdown (not a time), but keep histogram

    # ----------------------------
    # Process Lifecycle Logging
    # ----------------------------

    def log_process_start(self, inputs: Dict[str, Any]) -> None:
        """Log process start with input inspection"""
        if not self._should_log('DEBUG'):
            return

        with self._lock:
            self.total_processes += 1
            self._current_process_id = f"p#{self.total_processes}"
            # reset per-process breakdown
            self.calculation_breakdown = {}

        self._log("DEBUG", "─" * 60, "PROCESS_START")
        self._log("DEBUG", f"Starting reward calculation {self._current_process_id}", "PROCESS_START")

        input_summary = {
            'keys_provided': list(inputs.keys()),
            'step_idx': inputs.get('step_idx', 'not_provided'),
            'has_actions': 'actions' in inputs,
            'has_reward_inputs': 'reward_inputs' in inputs,
        }
        self._log("DEBUG", f"Input summary: {self._format_dict(input_summary)}", "INPUT_INSPECTION")

    def log_extracted_data(self, reward_data: Optional[Dict[str, Any]]) -> None:
        """Log extracted data with validation"""
        if not self._should_log('DEBUG'):
            return

        if not reward_data:
            self._log("ERROR", "NO DATA EXTRACTED - All bus reads failed!", "DATA_EXTRACTION")
            return

        # Check data quality
        data_quality = reward_data.get('data_quality', 'unknown')
        missing_keys = reward_data.get('missing_keys', []) or []
        broken_keys = reward_data.get('broken_keys', []) or []

        self._log("INFO", f"Data extraction completed - Quality: {str(data_quality).upper()}", "DATA_EXTRACTION")

        # Log missing keys
        if missing_keys:
            with self._lock:
                self.missing_keys_history.append({'timestamp': time.time(), 'keys': list(missing_keys)})
            self._log("WARNING", f"MISSING KEYS ({len(missing_keys)}): {', '.join(map(str, missing_keys))}", "MISSING_DATA")

            for key in missing_keys:
                with self._lock:
                    self.key_miss_counts[key] += 1
                expected = self.expected_keys.get(key, {})
                self._log(
                    "DEBUG",
                    f"  • {key}: Expected from {expected.get('source', 'Unknown')} "
                    f"(Required: {expected.get('required', False)})",
                    "MISSING_DETAIL"
                )

        # Log broken keys
        if broken_keys:
            with self._lock:
                self.broken_data_history.append({'timestamp': time.time(), 'keys': list(broken_keys)})
            self._log("WARNING", f"BROKEN DATA KEYS ({len(broken_keys)}): {', '.join(map(str, broken_keys))}", "BROKEN_DATA")

            for key in broken_keys:
                with self._lock:
                    self.key_error_counts[key] += 1

        # Log successful extractions
        extracted_keys = [
            k for k in reward_data.keys()
            if k not in ['data_quality', 'missing_keys', 'broken_keys', 'timestamp', 'validation_errors', 'fallbacks_used']
        ]
        if self._should_log('TRACE'):
            self._log("TRACE", f"Successfully extracted: {', '.join(map(str, extracted_keys))}", "EXTRACTION_SUCCESS")

        # Log key data points
        self._log_data_summary(reward_data)

    def _log_data_summary(self, data: Dict[str, Any]) -> None:
        """Log summary of extracted data"""
        if not self._should_log('DEBUG'):
            return

        summary: List[str] = []

        # Trades
        trades = data.get('trades', []) or []
        summary.append(f"Trades: {len(trades)}")
        if trades:
            try:
                total_pnl = float(sum(float(t.get('pnl', 0) or 0.0) for t in trades if isinstance(t, dict)))
            except Exception:
                total_pnl = 0.0
            summary.append(f"Total PnL: {total_pnl:.2f}")

        # Balance info
        risk_metrics = data.get('risk_metrics', {}) or {}
        balance = risk_metrics.get('balance', risk_metrics.get('equity', 'not_found'))
        summary.append(f"Balance: {balance}")

        drawdown = risk_metrics.get('current_drawdown', risk_metrics.get('drawdown', 'not_found'))
        summary.append(f"Drawdown: {drawdown}")

        # Market info
        regime = str(data.get('regime', 'unknown'))
        volatility = str(data.get('volatility_level', 'unknown'))
        summary.append(f"Regime: {regime.upper()}")
        summary.append(f"Volatility: {volatility.upper()}")

        self._log("INFO", "Data Summary: " + " | ".join(summary), "DATA_SUMMARY")

    def log_calculation_result(self, result: Dict[str, Any]) -> None:
        """Log reward calculation result"""
        if not self._should_log('DEBUG'):
            return

        reward = float(result.get('shaped_reward', 0.0))
        components = result.get('reward_components', {}) or {}

        self._log("INFO", f"Reward calculated: {reward:.4f}", "CALCULATION_RESULT")

        # Log component breakdown
        if self._should_log('DEBUG'):
            self._log_component_breakdown(components)

        # Track reward
        with self._lock:
            self.reward_history.append({
                'timestamp': time.time(),
                'reward': reward,
                'pnl': float(components.get('pnl', 0.0) or 0.0)
            })

    def _log_component_breakdown(self, components: Dict[str, Any]) -> None:
        """Log detailed component breakdown"""
        self._log("DEBUG", "Component Breakdown:", "COMPONENTS")

        penalties: Dict[str, float] = {}
        bonuses: Dict[str, float] = {}
        other: Dict[str, Any] = {}

        for key, value in components.items():
            if isinstance(value, (int, float)):
                if 'penalty' in key:
                    penalties[key] = float(value)
                elif 'bonus' in key:
                    bonuses[key] = float(value)
                else:
                    other[key] = value
            else:
                other[key] = value

        if penalties:
            self._log("DEBUG", "  Penalties:", "PENALTIES")
            for key, value in penalties.items():
                if abs(value) > 0.001:
                    self._log("DEBUG", f"    • {key}: {value:.4f}", "PENALTY")

        if bonuses:
            self._log("DEBUG", "  Bonuses:", "BONUSES")
            for key, value in bonuses.items():
                if abs(value) > 0.001:
                    self._log("DEBUG", f"    • {key}: {value:.4f}", "BONUS")

        if other and self._should_log('TRACE'):
            self._log("TRACE", "  Other:", "OTHER")
            shown = 0
            for key, value in other.items():
                if shown >= 15:  # avoid spam
                    self._log("TRACE", "    • ... more fields omitted", "OTHER")
                    break
                self._log("TRACE", f"    • {key}: {value}", "OTHER")
                shown += 1

    def log_process_complete(self, output: Dict[str, Any], duration: float) -> None:
        """Log process completion"""
        if not self._should_log('DEBUG'):
            return

        with self._lock:
            self.successful_processes += 1

        reward = float(output.get('shaped_reward', {}).get('reward', 0.0) if isinstance(output.get('shaped_reward'), dict) else output.get('shaped_reward', 0.0))
        success = bool(output.get('success', False))

        self._log("INFO", f"Process {self._current_process_id or ''} completed in {duration*1000:.2f}ms - Reward: {reward:.4f} - Success: {success}", "PROCESS_COMPLETE")

        # Log performance breakdown if in TRACE mode
        if self._should_log('TRACE') and self.calculation_breakdown:
            self._log("TRACE", "Performance Breakdown:", "PERFORMANCE")
            # Sort by time descending
            for operation, time_ms in sorted(self.calculation_breakdown.items(), key=lambda kv: kv[1], reverse=True):
                self._log("TRACE", f"  • {operation}: {time_ms:.2f}ms", "TIMING")

        self._log("DEBUG", "─" * 60, "PROCESS_END")

    # ----------------------------
    # Missing Data Analysis
    # ----------------------------

    def log_missing_data(self, data: Dict[str, Any]) -> None:
        """Log detailed missing data analysis"""
        if not self._should_log('WARNING'):
            return

        self._log("WARNING", "═" * 60, "MISSING_DATA_ANALYSIS")

        missing_keys = data.get('missing_keys', []) or []
        validation_errors = data.get('validation_errors', {}) or {}

        self._log("WARNING", f"Data extraction failed - {len(missing_keys)} keys missing", "DATA_FAILURE")

        # Categorize missing keys
        required_missing: List[str] = []
        optional_missing: List[str] = []

        for key in missing_keys:
            expected = self.expected_keys.get(key, {})
            if expected.get('required', False):
                required_missing.append(key)
            else:
                optional_missing.append(key)

        # Log required missing
        if required_missing:
            self._log("ERROR", f"CRITICAL - Required keys missing: {', '.join(required_missing)}", "REQUIRED_MISSING")
            for key in required_missing:
                expected = self.expected_keys.get(key, {})
                self._log("ERROR", f"  • {key}: Should be provided by {expected.get('source', 'Unknown')}", "MISSING_SOURCE")
                self._log_resolution_suggestion(key)

        # Log optional missing
        if optional_missing and self._should_log('DEBUG'):
            self._log("DEBUG", f"Optional keys missing: {', '.join(optional_missing)}", "OPTIONAL_MISSING")

        # Log validation errors
        if validation_errors:
            self._log("WARNING", "Validation errors found:", "VALIDATION_ERRORS")
            for key, error in validation_errors.items():
                self._log("WARNING", f"  • {key}: {error}", "VALIDATION_ERROR")

        self._log("WARNING", "═" * 60, "END_MISSING_ANALYSIS")

    def _log_resolution_suggestion(self, key: str) -> None:
        """Log suggestions for resolving missing keys"""
        suggestions = {
            'trade_data': "Check if TradingSystem module is running and publishing to bus",
            'risk_metrics': "Ensure RiskManager is initialized and processing",
            'market_context': "Verify MarketAnalyzer is active and has market data",
            'market_regime': "Check if RegimeDetector is enabled in configuration",
            'performance_data': "PerformanceTracker may not be initialized",
            'mistake_memory': "MistakeMemory module might be disabled",
        }
        suggestion = suggestions.get(key, "Check module initialization and bus registration")
        self._log("INFO", f"    → Suggestion: {suggestion}", "RESOLUTION_HINT")

    # ----------------------------
    # Error Tracking
    # ----------------------------

    def log_error(self, context: str, error: Exception, inputs: Optional[Dict] = None) -> None:
        """Log detailed error information"""
        self._log("ERROR", f"ERROR in {context}: {error!s}", "ERROR")

        # Track error
        error_entry = {
            'timestamp': time.time(),
            'context': context,
            'error_type': type(error).__name__,
            'error_msg': str(error),
            'traceback': traceback.format_exc()
        }
        with self._lock:
            self.error_history.append(error_entry)
            self.error_patterns[type(error).__name__] += 1

        # Log traceback in DEBUG mode
        if self._should_log('DEBUG'):
            self._log("DEBUG", f"Traceback:\n{error_entry['traceback']}", "TRACEBACK")

        # Log inputs if available
        if inputs and self._should_log('TRACE'):
            self._log("TRACE", f"Inputs at error: {self._format_dict(inputs)}", "ERROR_INPUTS")

    def log_error_details(self, error: Exception, processing_time: float) -> None:
        """Log comprehensive error details"""
        with self._lock:
            self.failed_processes += 1

        self._log("ERROR", "═" * 60, "ERROR_DETAILS")
        self._log("ERROR", f"Processing failed after {processing_time:.2f}ms", "FAILURE")

        error_type = type(error).__name__
        with self._lock:
            occurrences = self.error_patterns[error_type]

        self._log("ERROR", f"Error type: {error_type} (Occurrence #{occurrences})", "ERROR_PATTERN")

        # Log recent errors of same type
        if occurrences > 1:
            with self._lock:
                recent_similar = [e for e in self.error_history if e['error_type'] == error_type][-3:]
            self._log("WARNING", f"This error has occurred {occurrences} times", "RECURRING_ERROR")
            if self._should_log('DEBUG'):
                for err in recent_similar:
                    timestamp = datetime.fromtimestamp(err['timestamp']).strftime('%H:%M:%S')
                    self._log("DEBUG", f"  • {timestamp}: {err['context']}", "ERROR_HISTORY")

        self._log("ERROR", "═" * 60, "END_ERROR_DETAILS")

    # ----------------------------
    # State and Health Logging
    # ----------------------------

    def log_initialization_state(self) -> None:
        """Log initial system state"""
        if not self._should_log('INFO'):
            return

        self._log("INFO", "Reward system initialized - Debug mode ACTIVE", "INITIALIZATION")
        self._log("INFO", "Checking SmartInfoBus connectivity...", "BUS_CHECK")

        for key, info in self.expected_keys.items():
            try:
                value = self.smart_bus.get(key, "RewardDebug") if self.smart_bus else None
                if value is not None:
                    self._log("DEBUG", f"  ✓ {key}: Available", "BUS_KEY")
                else:
                    self._log("WARNING", f"  ✗ {key}: Not available (from {info['source']})", "BUS_KEY")
            except Exception as e:
                self._log("ERROR", f"  ✗ {key}: Error reading ({e!s})", "BUS_ERROR")

    def log_health_status(self, health: Dict[str, Any]) -> None:
        """Log health status update"""
        if not self._should_log('DEBUG'):
            return

        status = str(health.get('status', 'unknown'))

        if status != 'healthy' or self._should_log('TRACE'):
            self._log("INFO" if status == 'healthy' else "WARNING", f"Health Status: {status.upper()}", "HEALTH")

            if self._should_log('DEBUG'):
                details = []
                details.append(f"Circuit Breaker: {health.get('circuit_breaker', 'unknown')}")
                details.append(f"Mode: {health.get('current_mode', 'unknown')}")
                try:
                    details.append(f"Quality: {float(health.get('reward_quality', 0)):.3f}")
                except Exception:
                    details.append("Quality: n/a")
                try:
                    details.append(f"Win Rate: {float(health.get('win_rate', 0)):.1%}")
                except Exception:
                    details.append("Win Rate: n/a")
                for detail in details:
                    self._log("DEBUG", f"  • {detail}", "HEALTH_DETAIL")

    def log_monitoring_cycle(self) -> None:
        """Log monitoring cycle execution"""
        if not self._should_log('TRACE'):
            return

        self._log("TRACE", "Monitoring cycle executed", "MONITORING")

        # Log key statistics
        with self._lock:
            miss_items = list(self.key_miss_counts.items())
        if miss_items:
            top_missing = sorted(miss_items, key=lambda x: x[1], reverse=True)[:3]
            self._log("TRACE", f"Top missing keys: {', '.join([f'{k}({v})' for k, v in top_missing])}", "KEY_STATS")

    # ----------------------------
    # Specialized Logging
    # ----------------------------

    def log_confidence_calculation(
        self,
        confidence: float,
        action: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> None:
        """Log confidence calculation details"""
        if not self._should_log('DEBUG'):
            return
        self._log("DEBUG", f"Confidence calculated: {float(confidence):.3f}", "CONFIDENCE")

    def log_action_proposal(self, proposal: Dict[str, Any]) -> None:
        """Log action proposal"""
        if not self._should_log('DEBUG'):
            return

        recommendations = proposal.get('recommendations', []) or []
        self._log("INFO", f"Action proposal generated with {len(recommendations)} recommendations", "ACTION_PROPOSAL")

        if self._should_log('DEBUG'):
            for rec in recommendations[:3]:  # Log top 3
                self._log(
                    "DEBUG",
                    f"  • {rec.get('action', 'unknown')}: {rec.get('reason', 'no reason')} "
                    f"[{str(rec.get('priority', 'normal')).upper()}]",
                    "RECOMMENDATION"
                )

    def log_bus_updates(self, keys: List[str]) -> None:
        """Log bus update operations"""
        if not self._should_log('TRACE'):
            return
        self._log("TRACE", f"Updated bus keys: {', '.join(map(str, keys))}", "BUS_UPDATE")

    def log_reset(self) -> None:
        """Log system reset"""
        if not self._should_log('INFO'):
            return
        self._log("INFO", "System RESET - All state cleared", "RESET")

    def log_state_retrieval(self, state: Dict[str, Any]) -> None:
        """Log state retrieval"""
        if not self._should_log('TRACE'):
            return
        self._log("TRACE", f"State retrieved with {len(state)} components", "STATE_GET")

    def log_state_setting(self, state: Dict[str, Any]) -> None:
        """Log state setting"""
        if not self._should_log('DEBUG'):
            return
        self._log("DEBUG", f"State set with {len(state)} components", "STATE_SET")

    def log_shutdown(self) -> None:
        """Log system shutdown"""
        if not self._should_log('INFO'):
            return
        self._log("INFO", "System SHUTDOWN initiated", "SHUTDOWN")
        self._log_final_statistics()

    def increment_fallback(self) -> None:
        """Public helper to count a fallback event."""
        with self._lock:
            self.fallback_processes += 1

    # ----------------------------
    # Reports and Statistics
    # ----------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get debug statistics"""
        with self._lock:
            success_rate = (self.successful_processes / max(1, self.total_processes))
            op_summaries = {
                name: self._summarize_timings(list(samples))
                for name, samples in self.operation_timings.items()
                if samples
            }
            recent_rewards = list(self.reward_history)[-10:] if self.reward_history else []

            return {
                'enabled': self.enabled,
                'level': self.level,
                'total_processes': self.total_processes,
                'successful_processes': self.successful_processes,
                'failed_processes': self.failed_processes,
                'fallback_processes': self.fallback_processes,
                'success_rate': success_rate,
                'key_miss_counts': dict(self.key_miss_counts),
                'key_error_counts': dict(self.key_error_counts),
                'error_patterns': dict(self.error_patterns),
                'recent_rewards': recent_rewards,
                'timing_summaries': op_summaries,
            }

    def get_debug_report(self) -> str:
        """Generate comprehensive debug report"""
        stats = self.get_statistics()

        report = [
            "\n" + "═" * 60,
            "DEBUG REPORT",
            "═" * 60,
            f"Debug Level: {stats['level']}",
            f"Total Processes: {stats['total_processes']}",
            f"Success Rate: {stats['success_rate']:.1%}",
            f"Failed: {stats['failed_processes']}",
            f"Fallbacks: {stats['fallback_processes']}",
            ""
        ]

        # Key statistics
        if stats['key_miss_counts']:
            report.append("Most Missed Keys:")
            for key, count in sorted(stats['key_miss_counts'].items(), key=lambda x: x[1], reverse=True)[:5]:
                report.append(f"  • {key}: {count} misses")
            report.append("")

        # Error patterns
        if stats['error_patterns']:
            report.append("Error Patterns:")
            for error_type, count in stats['error_patterns'].items():
                report.append(f"  • {error_type}: {count} occurrences")
            report.append("")

        # Timing summaries
        if stats['timing_summaries']:
            report.append("Timing Summaries (ms):")
            # Show top 6 by mean time
            top_ops = sorted(stats['timing_summaries'].items(), key=lambda kv: kv[1]['mean'], reverse=True)[:6]
            for name, summary in top_ops:
                report.append(
                    f"  • {name}: mean={summary['mean']:.2f}, p95={summary['p95']:.2f}, max={summary['max']:.2f}, n={summary['n']}"
                )
            report.append("")

        # Recent performance
        recent = stats['recent_rewards'][-5:] if stats['recent_rewards'] else []
        if recent:
            rewards = [float(r['reward']) for r in recent if 'reward' in r]
            if rewards:
                avg_reward = float(np.mean(rewards))
                report.append(f"Recent Avg Reward: {avg_reward:.4f}")

        report.append("═" * 60)
        return "\n".join(report)

    def _log_final_statistics(self) -> None:
        """Log final statistics on shutdown"""
        stats = self.get_statistics()

        self._log("INFO", "Final Statistics:", "FINAL_STATS")
        self._log("INFO", f"  • Total processes: {stats['total_processes']}", "STAT")
        self._log("INFO", f"  • Success rate: {stats['success_rate']:.1%}", "STAT")

        if stats['key_miss_counts']:
            top_miss = max(stats['key_miss_counts'].items(), key=lambda x: x[1])
            self._log("INFO", f"  • Most missed key: {top_miss[0]} ({top_miss[1]} times)", "STAT")

        if stats['error_patterns']:
            top_error = max(stats['error_patterns'].items(), key=lambda x: x[1])
            self._log("INFO", f"  • Most common error: {top_error[0]} ({top_error[1]} times)", "STAT")

        # Top slow operations
        timing = stats.get('timing_summaries', {})
        if timing:
            top_slow = sorted(timing.items(), key=lambda kv: kv[1]['p95'], reverse=True)[:3]
            self._log("INFO", "  • Slowest operations (by p95):", "STAT")
            for name, s in top_slow:
                self._log("INFO", f"      - {name}: p95={s['p95']:.2f}ms (mean={s['mean']:.2f}ms, n={s['n']})", "STAT")

    # ----------------------------
    # Helper Methods
    # ----------------------------

    def _should_log(self, level: str) -> bool:
        """Check if should log at given level"""
        return self.enabled and self.levels.get(level, 5) >= self.current_level

    def _log(self, level: str, message: str, context: str = "", rate_limit_sec: Optional[float] = None) -> None:
        """Internal logging method with optional rate limiting"""
        if not self._should_log(level):
            return

        # Simple rate-limiting by (level, context)
        if rate_limit_sec is not None:
            key = (level, context)
            now = time.time()
            last = self._last_log_time.get(key, 0.0)
            if now - last < rate_limit_sec:
                return
            self._last_log_time[key] = now

        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        pid = f"[{self._current_process_id}] " if self._current_process_id else ""

        if context:
            formatted = f"[{level:7}] {timestamp} {pid}[{context:20}] {message}"
        else:
            formatted = f"[{level:7}] {timestamp} {pid}{message}"

        # Log using provided logger
        if self.logger:
            try:
                if level == 'ERROR':
                    self.logger.error(formatted)
                elif level == 'WARNING':
                    self.logger.warning(formatted)
                else:
                    self.logger.info(formatted)
            except Exception:
                # Fallback to stdout if the provided logger fails
                print(formatted)
        else:
            print(formatted)

    def _format_dict(self, d: Dict[str, Any], max_items: int = 10) -> str:
        """Format dictionary for logging"""
        if not d:
            return "{}"

        items = list(d.items())[:max_items]
        formatted: List[str] = []

        for key, value in items:
            try:
                if isinstance(value, float):
                    formatted.append(f"{key}: {value:.4f}")
                elif isinstance(value, (int, np.integer)):
                    formatted.append(f"{key}: {int(value)}")
                elif isinstance(value, str):
                    val = value.replace("\n", " ")
                    formatted.append(f"{key}: '{val[:50]}'" + ("…" if len(val) > 50 else ""))
                elif isinstance(value, (list, tuple)):
                    formatted.append(f"{key}: [{len(value)} items]")
                elif isinstance(value, dict):
                    formatted.append(f"{key}: {{...}}")
                else:
                    formatted.append(f"{key}: {type(value).__name__}")
            except Exception:
                formatted.append(f"{key}: <unprintable>")

        if len(d) > max_items:
            formatted.append(f"... +{len(d) - max_items} more")

        return "{" + ", ".join(formatted) + "}"

    @staticmethod
    def _summarize_timings(samples: List[float]) -> Dict[str, Any]:
        """Return summary stats (ms) for a list of timing samples."""
        if not samples:
            return {'n': 0, 'mean': 0.0, 'p95': 0.0, 'max': 0.0}
        arr = np.array(samples, dtype=np.float64)
        n = int(arr.size)
        mean = float(arr.mean())
        p95 = float(np.percentile(arr, 95))
        max_v = float(arr.max())
        return {'n': n, 'mean': mean, 'p95': p95, 'max': max_v}
