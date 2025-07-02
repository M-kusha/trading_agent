# envs/shared_utils.py
"""
Enhanced shared utilities with InfoBus integration and operator-centric design
Provides centralized infrastructure for InfoBus-integrated trading environment
"""
import time
import logging
import datetime
import numpy as np
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from pathlib import Path

# Import InfoBus infrastructure
from modules.utils.info_bus import (
    InfoBus, InfoBusBuilder, InfoBusExtractor, InfoBusUpdater, 
    extract_standard_context, create_info_bus, validate_info_bus,
    now_utc
)
from modules.utils.audit_utils import (
    RotatingLogger, AuditTracker, format_operator_message, system_audit
)
from modules.core.core import Module, ModuleConfig


# ═══════════════════════════════════════════════════════════════════
# Performance Profiling with InfoBus Integration  
# ═══════════════════════════════════════════════════════════════════

def profile_method(func):
    """Enhanced performance profiling with InfoBus logging"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        method_name = func.__name__
        
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            
            # Log slow operations with operator-friendly messages
            if elapsed > 0.1 and hasattr(self, 'logger'):
                self.logger.warning(
                    format_operator_message(
                        "⏱️", "SLOW_OPERATION", 
                        details=f"{method_name} took {elapsed:.3f}s",
                        context="performance_monitoring"
                    )
                )
                
                # Update InfoBus if available
                if hasattr(self, 'info_bus') and self.info_bus:
                    InfoBusUpdater.add_alert(
                        self.info_bus,
                        f"Slow operation: {method_name} ({elapsed:.3f}s)",
                        severity="warning",
                        module=self.__class__.__name__
                    )
            
            return result
            
        except Exception as e:
            elapsed = time.perf_counter() - start
            if hasattr(self, 'logger'):
                self.logger.error(
                    format_operator_message(
                        "💥", "METHOD_ERROR",
                        details=f"{method_name} failed after {elapsed:.3f}s: {e}",
                        context="error_tracking"
                    )
                )
            raise
            
    return wrapper


# ═══════════════════════════════════════════════════════════════════
# Enhanced Trading Pipeline with InfoBus Integration (Keeping Original Name)
# ═══════════════════════════════════════════════════════════════════

class TradingPipeline:
    """Enhanced trading pipeline with full InfoBus integration"""

    def __init__(self, modules: List[Module], config: Optional[Dict] = None):
        self.modules = modules
        self.config = config or {}
        self._module_map = {m.__class__.__name__: m for m in modules}
        
        # Pipeline state
        self.expected_size: Optional[int] = None
        self.step_count = 0
        self.last_info_bus: Optional[InfoBus] = None
        
        # Enhanced logging
        self.logger = RotatingLogger(
            "InfoBusPipeline",
            "logs/pipeline/trading_pipeline.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("TradingPipeline")
        
        # Performance tracking
        self.module_performance = defaultdict(lambda: {
            'total_time': 0.0,
            'call_count': 0,
            'error_count': 0,
            'avg_time': 0.0
        })
        
        self.logger.info(
            format_operator_message(
                "🔧", "PIPELINE_INIT",
                details=f"Initialized with {len(modules)} modules",
                context="system_startup"
            )
        )

    def reset(self):
        """Enhanced reset with InfoBus coordination"""
        self.step_count = 0
        self.last_info_bus = None
        
        for module in self.modules:
            try:
                module.reset()
            except Exception as e:
                self.logger.error(
                    format_operator_message(
                        "🔄", "MODULE_RESET_FAILED",
                        instrument=module.__class__.__name__,
                        details=str(e),
                        context="pipeline_reset"
                    )
                )
        
        self.logger.info(
            format_operator_message(
                "✅", "PIPELINE_RESET", 
                details="All modules reset successfully",
                context="system_state"
            )
        )

    @profile_method
    def step(self, info_bus: InfoBus) -> np.ndarray:
        """Enhanced pipeline step with comprehensive InfoBus processing"""
        self.step_count += 1
        self.last_info_bus = info_bus
        
        # Validate InfoBus
        quality = validate_info_bus(info_bus)
        if not quality.is_valid:
            self.logger.warning(
                format_operator_message(
                    "⚠️", "INVALID_INFO_BUS",
                    details=f"Quality issues: {quality.issues}",
                    context="data_validation"
                )
            )
        
        # Extract standard context for all modules
        context = extract_standard_context(info_bus)
        
        # Process modules and collect observations
        obs_parts: List[np.ndarray] = []
        module_errors = []
        
        for module in self.modules:
            module_name = module.__class__.__name__
            
            # Check if module is enabled
            if not self._is_module_enabled(module_name, info_bus):
                continue
                
            start_time = time.perf_counter()
            
            try:
                # Process module with InfoBus
                self._process_module_step(module, info_bus, context)
                
                # Get observation components
                obs_component = module.get_observation_components()
                obs_component = self._sanitize_observation_component(
                    obs_component, module_name
                )
                obs_parts.append(obs_component)
                
                # Update performance metrics
                elapsed = time.perf_counter() - start_time
                self._update_module_performance(module_name, elapsed, success=True)
                
            except Exception as e:
                # Handle module errors gracefully
                elapsed = time.perf_counter() - start_time
                self._handle_module_error(module_name, e, elapsed)
                module_errors.append((module_name, str(e)))
                
                # Add empty observation to maintain consistency
                obs_parts.append(np.zeros(0, dtype=np.float32))
        
        # Log any module errors
        if module_errors:
            self._log_module_errors(module_errors, info_bus)
        
        # Combine observations
        obs = self._combine_observations(obs_parts)
        
        # Update InfoBus with pipeline results
        self._update_info_bus_with_pipeline_results(info_bus, obs, module_errors)
        
        return obs

    def _is_module_enabled(self, module_name: str, info_bus: InfoBus) -> bool:
        """Check if module is enabled via InfoBus or config"""
        # Check InfoBus for dynamic enablement
        module_config = info_bus.get('module_config', {})
        if module_name in module_config:
            return module_config[module_name].get('enabled', True)
        
        # Check environment config
        env = info_bus.get('env')
        if env and hasattr(env, 'module_enabled'):
            return env.module_enabled.get(module_name, True)
        
        return True

    def _process_module_step(self, module: Module, info_bus: InfoBus, context: Dict[str, Any]):
        """Process individual module step with proper InfoBus handling"""
        module_name = module.__class__.__name__
        
        # Special handling for memory modules that don't take InfoBus directly
        if module_name in ['MistakeMemory', 'HistoricalReplayAnalyzer', 
                          'PlaybookMemory', 'MemoryBudgetOptimizer']:
            # These modules are fed by environment directly - just get observations
            return
        
        # Most modules now take InfoBus as primary input
        if hasattr(module.step, '__code__'):
            sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
            
            if 'info_bus' in sig:
                # New InfoBus-integrated modules
                module.step(info_bus=info_bus)
            else:
                # Legacy modules - extract required parameters
                kwargs = {}
                for param in sig:
                    if param in info_bus:
                        kwargs[param] = info_bus[param]
                    elif param in context:
                        kwargs[param] = context[param]
                
                module.step(**kwargs)

    def _sanitize_observation_component(self, obs: np.ndarray, module_name: str) -> np.ndarray:
        """Sanitize observation component from module"""
        # Ensure 1D float32
        obs = np.asarray(obs, dtype=np.float32).ravel()
        
        # Handle NaN/Inf values
        if not np.all(np.isfinite(obs)):
            self.logger.warning(
                format_operator_message(
                    "🧹", "SANITIZE_OBSERVATION",
                    instrument=module_name,
                    details="Replaced NaN/Inf values with zeros",
                    context="data_cleaning"
                )
            )
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        return obs

    def _combine_observations(self, obs_parts: List[np.ndarray]) -> np.ndarray:
        """Combine observation parts with size management"""
        # Concatenate all parts
        obs = np.concatenate(obs_parts) if obs_parts else np.zeros(0, dtype=np.float32)
        
        # Handle size consistency
        if self.expected_size is None:
            self.expected_size = obs.size
            self.logger.info(
                format_operator_message(
                    "📐", "OBSERVATION_SIZE_LOCKED",
                    details=f"Pipeline observation size: {self.expected_size}",
                    context="system_configuration"
                )
            )
        else:
            # Ensure consistent size
            if obs.size != self.expected_size:
                if obs.size < self.expected_size:
                    # Pad with zeros
                    pad_size = self.expected_size - obs.size
                    obs = np.concatenate([obs, np.zeros(pad_size, dtype=np.float32)])
                else:
                    # Truncate
                    obs = obs[:self.expected_size]
                    
                self.logger.debug(
                    f"Adjusted observation size from {obs.size} to {self.expected_size}"
                )
        
        return obs

    def _handle_module_error(self, module_name: str, error: Exception, elapsed: float):
        """Handle module processing errors"""
        self._update_module_performance(module_name, elapsed, success=False)
        
        self.logger.error(
            format_operator_message(
                "💥", "MODULE_ERROR",
                instrument=module_name,
                details=f"Processing failed: {error}",
                result=f"Skipped module after {elapsed:.3f}s",
                context="error_recovery"
            )
        )

    def _log_module_errors(self, module_errors: List[Tuple[str, str]], info_bus: InfoBus):
        """Log module errors to InfoBus and audit system"""
        if not module_errors:
            return
            
        error_summary = f"{len(module_errors)} modules failed: " + ", ".join(
            f"{name}" for name, _ in module_errors
        )
        
        # Add to InfoBus
        InfoBusUpdater.add_alert(
            info_bus,
            error_summary,
            severity="error",
            module="Pipeline"
        )
        
        # Record audit event
        self.audit_tracker.record_event(
            "module_errors",
            "Pipeline",
            {"errors": module_errors, "step": self.step_count},
            severity="error"
        )

    def _update_module_performance(self, module_name: str, elapsed: float, success: bool):
        """Update module performance metrics"""
        perf = self.module_performance[module_name]
        perf['total_time'] += elapsed
        perf['call_count'] += 1
        if not success:
            perf['error_count'] += 1
        
        perf['avg_time'] = perf['total_time'] / max(perf['call_count'], 1)

    def _update_info_bus_with_pipeline_results(self, info_bus: InfoBus, obs: np.ndarray, 
                                             module_errors: List[Tuple[str, str]]):
        """Update InfoBus with pipeline processing results"""
        pipeline_data = {
            'step_count': self.step_count,
            'observation_size': obs.size,
            'module_count': len(self.modules),
            'error_count': len(module_errors),
            'performance': dict(self.module_performance)
        }
        
        InfoBusUpdater.add_module_data(info_bus, 'Pipeline', pipeline_data)

    def get_module(self, name: str) -> Optional[Module]:
        """Get module by name"""
        return self._module_map.get(name)

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        return {
            'total_steps': self.step_count,
            'module_count': len(self.modules),
            'expected_obs_size': self.expected_size,
            'module_performance': dict(self.module_performance)
        }


# ═══════════════════════════════════════════════════════════════════
# Enhanced Risk Management with InfoBus Integration (Keeping Original Name)
# ═══════════════════════════════════════════════════════════════════

class UnifiedRiskManager:
    """Enhanced centralized risk management with InfoBus integration"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.dd_limit = config.get('dd_limit', 0.3)
        self.correlation_limit = config.get('correlation_limit', 0.8)
        self.var_limit = config.get('var_limit', 0.1)
        self.max_positions = config.get('max_positions', 10)
        self.alert_cooldown = config.get('alert_cooldown', 5)
        
        # Enhanced logging
        self.logger = logger or RotatingLogger(
            "InfoBusRiskManager",
            "logs/risk/unified_risk_manager.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("RiskManager")
        
        # Risk state tracking
        self.last_alerts = defaultdict(int)
        self.risk_violations = 0
        self.last_risk_check = 0
        
        # Performance metrics
        self.risk_stats = {
            'checks_performed': 0,
            'violations_detected': 0,
            'trades_blocked': 0,
            'alerts_raised': 0
        }
        
        self.logger.info(
            format_operator_message(
                "🛡️", "RISK_MANAGER_INIT",
                details=f"DD={self.dd_limit:.1%}, Corr={self.correlation_limit:.2f}",
                context="risk_system_startup"
            )
        )
        
    def pre_trade_check(self, info_bus: InfoBus) -> Tuple[bool, str]:
        """Enhanced pre-trade risk checks with InfoBus integration"""
        self.risk_stats['checks_performed'] += 1
        
        # Extract risk context from InfoBus
        risk_context = InfoBusExtractor.extract_risk_context(info_bus)
        
        # Check drawdown
        current_dd = risk_context.get('drawdown_pct', 0.0)
        if current_dd > self.dd_limit:
            return self._handle_risk_violation(
                info_bus, "DRAWDOWN_LIMIT",
                f"Drawdown {current_dd:.1%} exceeds limit {self.dd_limit:.1%}"
            )
        
        # Check correlations
        correlations = info_bus.get('correlations', {})
        max_corr = max(correlations.values()) if correlations else 0
        if max_corr > self.correlation_limit:
            return self._handle_risk_violation(
                info_bus, "CORRELATION_LIMIT",
                f"Correlation {max_corr:.2f} exceeds limit {self.correlation_limit:.2f}"
            )
        
        # Check position count
        position_count = InfoBusExtractor.get_position_count(info_bus)
        if position_count >= self.max_positions:
            return self._handle_risk_violation(
                info_bus, "POSITION_LIMIT",
                f"Position count {position_count} exceeds limit {self.max_positions}"
            )
        
        # Check overall risk score
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        if risk_score > 0.8:  # High risk threshold
            return self._handle_risk_violation(
                info_bus, "RISK_SCORE",
                f"Risk score {risk_score:.2f} too high"
            )
        
        # All checks passed
        self._update_info_bus_risk_status(info_bus, "PASSED", "All risk checks passed")
        return True, "All risk checks passed"
    
    def _handle_risk_violation(self, info_bus: InfoBus, violation_type: str, 
                              message: str) -> Tuple[bool, str]:
        """Handle risk violation with proper logging and alerts"""
        self.risk_stats['violations_detected'] += 1
        self.risk_stats['trades_blocked'] += 1
        
        # Check alert cooldown
        current_step = info_bus.get('step_idx', 0)
        if current_step - self.last_alerts[violation_type] < self.alert_cooldown:
            return False, message  # Silent block
        
        self.last_alerts[violation_type] = current_step
        self.risk_stats['alerts_raised'] += 1
        
        # Log operator message
        self.logger.warning(
            format_operator_message(
                "🚫", "TRADE_BLOCKED",
                details=message,
                result="Trade execution prevented",
                context="risk_management"
            )
        )
        
        # Update InfoBus
        InfoBusUpdater.add_alert(info_bus, message, severity="warning", module="RiskManager")
        self._update_info_bus_risk_status(info_bus, "VIOLATION", message)
        
        # Record audit event
        self.audit_tracker.record_event(
            "risk_violation",
            "RiskManager", 
            {"type": violation_type, "message": message},
            severity="warning"
        )
        
        return False, message
    
    def post_trade_update(self, info_bus: InfoBus, trade: Dict[str, Any]):
        """Update risk systems after trade with InfoBus coordination"""
        # Extract trade info
        pnl = trade.get('pnl', 0.0)
        instrument = trade.get('instrument', 'UNKNOWN')
        
        # Log trade impact
        if abs(pnl) > 100:  # Significant trade
            severity = "info" if pnl > 0 else "warning"
            self.logger.info(
                format_operator_message(
                    "💰" if pnl > 0 else "💸", "SIGNIFICANT_TRADE",
                    instrument=instrument,
                    details=f"PnL: {pnl:+.2f}",
                    context="trade_monitoring"
                )
            )
        
        # Update InfoBus with trade impact
        trade_impact = {
            'instrument': instrument,
            'pnl': pnl,
            'timestamp': now_utc(),
            'risk_impact': self._assess_trade_risk_impact(trade)
        }
        
        InfoBusUpdater.add_module_data(info_bus, 'RiskManager', {
            'last_trade': trade_impact,
            'stats': self.risk_stats
        })
    
    def _assess_trade_risk_impact(self, trade: Dict[str, Any]) -> str:
        """Assess the risk impact of a trade"""
        pnl = trade.get('pnl', 0.0)
        size = trade.get('size', 0.0)
        
        if abs(pnl) > 1000:
            return "HIGH"
        elif abs(pnl) > 500:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _update_info_bus_risk_status(self, info_bus: InfoBus, status: str, message: str):
        """Update InfoBus with current risk status"""
        risk_status = {
            'status': status,
            'message': message,
            'timestamp': now_utc(),
            'stats': self.risk_stats,
            'limits': {
                'drawdown': self.dd_limit,
                'correlation': self.correlation_limit,
                'positions': self.max_positions
            }
        }
        
        InfoBusUpdater.update_risk_snapshot(info_bus, risk_status)


# ═══════════════════════════════════════════════════════════════════
# Enhanced Environment Utilities
# ═══════════════════════