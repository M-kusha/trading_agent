# ─────────────────────────────────────────────────────────────
# File: modules/core/error_pinpointer.py
# 🚀 Enhanced error diagnosis and debugging for SmartInfoBus
# ─────────────────────────────────────────────────────────────

import traceback
import inspect
import sys
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import re
from pathlib import Path

import numpy as np

from modules.utils.info_bus import SmartInfoBus, InfoBusManager
from modules.utils.audit_utils import format_operator_message, RotatingLogger
from modules.core.module_orchestrator import ModuleOrchestrator


@dataclass
class ErrorContext:
    """Comprehensive error context"""
    error_type: str
    error_message: str
    module_name: str
    method_name: str
    line_number: int
    file_path: str
    timestamp: datetime
    traceback_full: str
    local_variables: Dict[str, Any]
    module_state: Dict[str, Any]
    infobus_snapshot: Dict[str, Any]
    related_errors: List['ErrorContext']
    
    def to_plain_english(self) -> str:
        """Convert error to plain English explanation"""
        explanation = f"""
ERROR REPORT
============
What Happened: {self.error_type} in {self.module_name}
When: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Where: {self.file_path}:{self.line_number}

SIMPLE EXPLANATION:
The module '{self.module_name}' encountered an error while executing '{self.method_name}'.
Error message: {self.error_message}

LIKELY CAUSE:
{self._diagnose_likely_cause()}

SUGGESTED FIX:
{self._suggest_fix()}
"""
        return explanation.strip()
    
    def _diagnose_likely_cause(self) -> str:
        """Diagnose likely cause based on error pattern"""
        error_msg = self.error_message.lower()
        
        if "nonetype" in error_msg:
            return "A required value was not found (None/null). Check if all required data is being provided."
        elif "key" in error_msg and "not found" in error_msg:
            return "Missing data field. The module expected data that wasn't available."
        elif "timeout" in error_msg:
            return "The module took too long to execute. Consider optimizing the code or increasing timeout."
        elif "connection" in error_msg:
            return "Network or connection issue. Check external service availability."
        elif "permission" in error_msg:
            return "Access denied. Check file permissions or API credentials."
        elif "memory" in error_msg:
            return "Out of memory. The module may be processing too much data."
        else:
            return "An unexpected error occurred. Check the full traceback for details."
    
    def _suggest_fix(self) -> str:
        """Suggest fix based on error type"""
        error_msg = self.error_message.lower()
        
        if "nonetype" in error_msg:
            return "Add null checks before accessing object attributes"
        elif "key" in error_msg:
            return "Use .get() method with default values instead of direct key access"
        elif "timeout" in error_msg:
            return "1. Optimize the slow operation\n2. Increase timeout in orchestration_policy.yaml\n3. Break operation into smaller chunks"
        elif "index" in error_msg:
            return "Check array bounds before accessing elements"
        else:
            return "Review the error traceback and module implementation"


class ErrorPinpointer:
    """
    Advanced error diagnosis system for SmartInfoBus modules.
    Provides detailed error analysis and plain English explanations.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator
        self.error_history: List[ErrorContext] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Setup logging
        self.logger = RotatingLogger(
            name="ErrorPinpointer",
            log_path="logs/errors/error_analysis.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Common error patterns
        self.known_patterns = self._load_known_patterns()
    
    def analyze_exception(self, exception: Exception, module_name: str = "Unknown",
                         method_name: str = "Unknown") -> ErrorContext:
        """Analyze an exception and create comprehensive error context"""
        # Get traceback
        tb = traceback.extract_tb(exception.__traceback__)
        
        # Find the most relevant frame (in module code, not framework)
        relevant_frame = self._find_relevant_frame(tb, module_name)
        
        # Extract local variables from frame
        local_vars = self._extract_local_variables(exception.__traceback__)
        
        # Get module state if available
        module_state = self._get_module_state(module_name)
        
        # Get InfoBus snapshot
        infobus_snapshot = self._get_infobus_snapshot()
        
        # Find related errors
        related = self._find_related_errors(exception, module_name)
        
        # Create error context
        context = ErrorContext(
            error_type=type(exception).__name__,
            error_message=str(exception),
            module_name=module_name,
            method_name=method_name,
            line_number=relevant_frame.lineno if relevant_frame else 0,
            file_path=relevant_frame.filename if relevant_frame else "Unknown",
            timestamp=datetime.now(),
            traceback_full=traceback.format_exc(),
            local_variables=local_vars,
            module_state=module_state,
            infobus_snapshot=infobus_snapshot,
            related_errors=related
        )
        
        # Record error
        self.error_history.append(context)
        self._update_error_patterns(context)
        
        # Log analysis
        self.logger.error(
            format_operator_message(
                "🔍", "ERROR ANALYZED",
                instrument=module_name,
                details=f"{context.error_type}: {context.error_message[:50]}...",
                context="debugging"
            )
        )
        
        return context
    
    def _find_relevant_frame(self, tb: List, module_name: str):
        """Find the most relevant traceback frame"""
        # Prefer frames from the specific module
        for frame in reversed(tb):
            if module_name.lower() in frame.filename.lower():
                return frame
        
        # Otherwise, skip framework frames
        for frame in reversed(tb):
            if not any(skip in frame.filename for skip in 
                      ['asyncio', 'concurrent', 'threading', 'gymnasium']):
                return frame
        
        # Last resort
        return tb[-1] if tb else None
    
    def _extract_local_variables(self, tb) -> Dict[str, Any]:
        """Extract local variables from traceback"""
        local_vars = {}
        
        if tb is None:
            return local_vars
        
        # Get the last frame
        while tb.tb_next is not None:
            tb = tb.tb_next
        
        frame = tb.tb_frame
        
        # Extract locals (be careful with large objects)
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
                else:
                    local_vars[var_name] = f"{type(var_value).__name__}"
            except:
                local_vars[var_name] = "<Unable to serialize>"
        
        return local_vars
    
    def _get_module_state(self, module_name: str) -> Dict[str, Any]:
        """Get current module state if available"""
        if not self.orchestrator:
            return {}
        
        module = self.orchestrator.get_module_by_name(module_name)
        if module and hasattr(module, 'get_state'):
            try:
                return module.get_state()
            except:
                return {"error": "Failed to get module state"}
        
        return {}
    
    def _get_infobus_snapshot(self) -> Dict[str, Any]:
        """Get snapshot of current InfoBus state"""
        smart_bus = InfoBusManager.get_instance()
        
        snapshot = {
            'data_keys': list(smart_bus._data_store.keys())[:20],  # First 20 keys
            'total_keys': len(smart_bus._data_store),
            'disabled_modules': list(smart_bus._module_disabled),
            'recent_events': list(smart_bus._event_log)[-10:],  # Last 10 events
            'performance_metrics': smart_bus.get_performance_metrics()
        }
        
        return snapshot
    
    def _find_related_errors(self, exception: Exception, 
                           module_name: str) -> List[ErrorContext]:
        """Find related errors in history"""
        related = []
        
        # Look for similar errors in recent history
        for error in self.error_history[-10:]:  # Last 10 errors
            if (error.module_name == module_name or 
                error.error_type == type(exception).__name__ or
                self._are_errors_similar(error.error_message, str(exception))):
                related.append(error)
        
        return related[:3]  # Max 3 related errors
    
    def _are_errors_similar(self, msg1: str, msg2: str) -> bool:
        """Check if two error messages are similar"""
        # Simple similarity check
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return False
        
        overlap = len(words1.intersection(words2))
        similarity = overlap / min(len(words1), len(words2))
        
        return similarity > 0.5
    
    def _update_error_patterns(self, context: ErrorContext):
        """Track error patterns for analysis"""
        pattern_key = f"{context.module_name}:{context.error_type}"
        self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
    
    def _load_known_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load known error patterns and solutions"""
        return {
            "KeyError": {
                "description": "Missing data key",
                "common_causes": [
                    "InfoBus data not set by provider module",
                    "Typo in key name",
                    "Module execution order issue"
                ],
                "solutions": [
                    "Check if provider module is enabled",
                    "Verify key name matches exactly",
                    "Check execution order in orchestrator"
                ]
            },
            "TimeoutError": {
                "description": "Module execution timeout",
                "common_causes": [
                    "Complex computation taking too long",
                    "External API slow response",
                    "Infinite loop"
                ],
                "solutions": [
                    "Increase timeout in orchestration_policy.yaml",
                    "Optimize algorithm",
                    "Add caching for expensive operations"
                ]
            },
            "AttributeError": {
                "description": "Attribute not found on object",
                "common_causes": [
                    "Object is None",
                    "Wrong object type",
                    "Attribute doesn't exist"
                ],
                "solutions": [
                    "Add None checks",
                    "Verify object type before access",
                    "Check attribute spelling"
                ]
            }
        }
    
    def generate_error_report(self, context: ErrorContext) -> str:
        """Generate comprehensive error report"""
        report = context.to_plain_english()
        
        # Add pattern analysis
        pattern_key = f"{context.module_name}:{context.error_type}"
        occurrences = self.error_patterns.get(pattern_key, 1)
        
        if occurrences > 1:
            report += f"\n\nPATTERN DETECTED:\nThis error has occurred {occurrences} times."
        
        # Add known pattern info
        if context.error_type in self.known_patterns:
            pattern_info = self.known_patterns[context.error_type]
            report += f"\n\nKNOWN ERROR TYPE: {pattern_info['description']}"
            report += f"\nCommon Causes:\n"
            for cause in pattern_info['common_causes']:
                report += f"  • {cause}\n"
            report += f"\nSuggested Solutions:\n"
            for solution in pattern_info['solutions']:
                report += f"  • {solution}\n"
        
        # Add local variable info if relevant
        if context.local_variables:
            report += f"\n\nRELEVANT VARIABLES:\n"
            for var, value in list(context.local_variables.items())[:5]:
                report += f"  {var} = {value}\n"
        
        return report
    
    def suggest_debugging_steps(self, context: ErrorContext) -> List[str]:
        """Suggest specific debugging steps"""
        steps = []
        
        # General steps
        steps.append(f"1. Check line {context.line_number} in {Path(context.file_path).name}")
        
        # Error-specific steps
        if "KeyError" in context.error_type:
            key_match = re.search(r"['\"]([^'\"]+)['\"]", context.error_message)
            if key_match:
                key = key_match.group(1)
                steps.append(f"2. Verify that '{key}' is being set in InfoBus")
                steps.append(f"3. Check module that should provide '{key}'")
                steps.append(f"4. Use smart_bus.get('{key}', default_value) instead")
        
        elif "timeout" in context.error_message.lower():
            steps.append("2. Profile the module to find slow operations")
            steps.append("3. Check for infinite loops or recursive calls")
            steps.append("4. Consider breaking operation into smaller chunks")
        
        elif "NoneType" in context.error_message:
            steps.append("2. Add null checks before the error line")
            steps.append("3. Trace back to find where None is coming from")
            steps.append("4. Set appropriate default values")
        
        # Module state debugging
        if context.module_state:
            steps.append(f"5. Module state available - check for inconsistencies")
        
        # Related errors
        if context.related_errors:
            steps.append(f"6. Found {len(context.related_errors)} related errors - check for patterns")
        
        return steps
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors"""
        summary = {
            'total_errors': len(self.error_history),
            'unique_modules': len(set(e.module_name for e in self.error_history)),
            'most_common_errors': [],
            'problem_modules': []
        }
        
        # Most common error types
        error_counts = {}
        module_errors = {}
        
        for error in self.error_history:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            module_errors[error.module_name] = module_errors.get(error.module_name, 0) + 1
        
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
        
        return summary
    
    def create_debug_snapshot(self, module_name: str) -> Dict[str, Any]:
        """Create a debugging snapshot for a module"""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'recent_errors': [],
            'module_state': self._get_module_state(module_name),
            'infobus_state': self._get_infobus_snapshot(),
            'execution_context': {}
        }
        
        # Get recent errors for this module
        for error in self.error_history[-20:]:
            if error.module_name == module_name:
                snapshot['recent_errors'].append({
                    'type': error.error_type,
                    'message': error.error_message,
                    'timestamp': error.timestamp.isoformat(),
                    'line': error.line_number
                })
        
        # Get execution context from orchestrator
        if self.orchestrator:
            smart_bus = InfoBusManager.get_instance()
            snapshot['execution_context'] = {
                'is_enabled': smart_bus.is_module_enabled(module_name),
                'failure_count': smart_bus._module_failures.get(module_name, 0),
                'avg_latency': np.mean(list(smart_bus._latency_history.get(module_name, []))) if module_name in smart_bus._latency_history else 0
            }
        
        return snapshot