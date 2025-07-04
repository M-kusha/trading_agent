# ─────────────────────────────────────────────────────────────
# File: envs/shared_utils.py (RISK MANAGEMENT FIX)
# 🔧 CRITICAL FIX: Risk score calculation should be 0-1, not 0-100
# ─────────────────────────────────────────────────────────────

import time
import logging
import datetime
import numpy as np
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple, Callable
from collections import defaultdict, deque
from pathlib import Path

from modules.utils.info_bus import (
    InfoBus, InfoBusBuilder, InfoBusExtractor, InfoBusUpdater, 
    extract_standard_context, create_info_bus, validate_info_bus,
    now_utc
)
from modules.utils.audit_utils import (
    RotatingLogger, AuditTracker, format_operator_message, system_audit
)
from modules.core.core import Module, ModuleConfig


def profile_method(func):
    """Enhanced performance profiling with InfoBus logging"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        method_name = func.__name__
        
        try:
            result = func(self, *args, **kwargs)
            elapsed = time.perf_counter() - start
            
            if elapsed > 0.1 and hasattr(self, 'logger'):
                self.logger.warning(
                    format_operator_message(
                        "⏱️", "SLOW_OPERATION", 
                        details=f"{method_name} took {elapsed:.3f}s",
                        context="performance_monitoring"
                    )
                )
                
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


class TradingPipeline:
    """Enhanced trading pipeline with full InfoBus integration"""

    def __init__(self, modules: List[Module], config: Optional[Dict] = None):
        self.modules = modules
        self.config = config or {}
        self._module_map = {m.__class__.__name__: m for m in modules}
        
        self.expected_size: Optional[int] = None
        self.step_count = 0
        self.last_info_bus: Optional[InfoBus] = None
        
        self.logger = RotatingLogger(
            "InfoBusPipeline",
            "logs/pipeline/trading_pipeline.log",
            max_lines=2000,
            operator_mode=True
        )
        
        self.audit_tracker = AuditTracker("TradingPipeline")
        
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
        
        quality = validate_info_bus(info_bus)
        if not quality.is_valid:
            self.logger.warning(
                format_operator_message(
                    "⚠️", "INVALID_INFO_BUS",
                    details=f"Quality issues: {len(quality.issues)}",
                    context="data_validation"
                )
            )
        
        context = extract_standard_context(info_bus)
        
        obs_parts: List[np.ndarray] = []
        module_errors = []
        
        for module in self.modules:
            module_name = module.__class__.__name__
            
            if not self._is_module_enabled(module_name, info_bus):
                continue
                
            start_time = time.perf_counter()
            
            try:
                self._process_module_step(module, info_bus, context)
                
                obs_component = module.get_observation_components()
                obs_component = self._sanitize_observation_component(
                    obs_component, module_name
                )
                obs_parts.append(obs_component)
                
                elapsed = time.perf_counter() - start_time
                self._update_module_performance(module_name, elapsed, success=True)
                
            except Exception as e:
                elapsed = time.perf_counter() - start_time
                self._handle_module_error(module_name, e, elapsed)
                module_errors.append((module_name, str(e)))
                
                obs_parts.append(np.zeros(0, dtype=np.float32))
        
        if module_errors:
            self._log_module_errors(module_errors, info_bus)
        
        obs = self._combine_observations(obs_parts)
        self._update_info_bus_with_pipeline_results(info_bus, obs, module_errors)
        
        return obs

    def _is_module_enabled(self, module_name: str, info_bus: InfoBus) -> bool:
        """Check if module is enabled via InfoBus or config"""
        module_config = info_bus.get('module_config', {})
        if module_name in module_config:
            return module_config[module_name].get('enabled', True)
        
        env = info_bus.get('env')
        if env and hasattr(env, 'module_enabled'):
            return env.module_enabled.get(module_name, True)
        
        return True

    def _process_module_step(self, module: Module, info_bus: InfoBus, context: Dict[str, Any]):
        """Process individual module step with proper InfoBus handling"""
        module_name = module.__class__.__name__
        
        if module_name in ['MistakeMemory', 'HistoricalReplayAnalyzer', 
                          'PlaybookMemory', 'MemoryBudgetOptimizer']:
            return
        
        if hasattr(module.step, '__code__'):
            sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
            
            if 'info_bus' in sig:
                module.step(info_bus=info_bus)
            else:
                kwargs = {}
                for param in sig:
                    if param in info_bus:
                        kwargs[param] = info_bus[param]
                    elif param in context:
                        kwargs[param] = context[param]
                
                module.step(**kwargs)

    def _sanitize_observation_component(self, obs: np.ndarray, module_name: str) -> np.ndarray:
        """Sanitize observation component from module"""
        obs = np.asarray(obs, dtype=np.float32).ravel()
        
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
        obs = np.concatenate(obs_parts) if obs_parts else np.zeros(0, dtype=np.float32)
        
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
            if obs.size != self.expected_size:
                if obs.size < self.expected_size:
                    pad_size = self.expected_size - obs.size
                    obs = np.concatenate([obs, np.zeros(pad_size, dtype=np.float32)])
                else:
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
        
        InfoBusUpdater.add_alert(
            info_bus,
            error_summary,
            severity="error",
            module="Pipeline"
        )
        
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
# 🔧 FIXED Risk Management with Correct Scoring (0-1 range)
# ═══════════════════════════════════════════════════════════════════

class UnifiedRiskManager:
    """🔧 FIXED: Enhanced centralized risk management with proper scoring"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.dd_limit = config.get('dd_limit', 0.3)
        self.correlation_limit = config.get('correlation_limit', 0.8)
        self.var_limit = config.get('var_limit', 0.1)
        self.max_positions = config.get('max_positions', 10)
        self.alert_cooldown = config.get('alert_cooldown', 5)
        
        # 🔧 FIX: Add risk score thresholds (0-1 range)
        self.risk_score_warning = config.get('risk_score_warning', 0.7)
        self.risk_score_critical = config.get('risk_score_critical', 0.8)
        
        self.logger = logger or RotatingLogger(
            "InfoBusRiskManager",
            "logs/risk/unified_risk_manager.log",
            max_lines=2000,
            operator_mode=True
        )
        
        self.audit_tracker = AuditTracker("RiskManager")
        
        self.last_alerts = defaultdict(int)
        self.risk_violations = 0
        self.last_risk_check = 0
        
        self.risk_stats = {
            'checks_performed': 0,
            'violations_detected': 0,
            'trades_blocked': 0,
            'alerts_raised': 0
        }
        
        self.logger.info(
            format_operator_message(
                "🛡️", "RISK_MANAGER_INIT",
                details=f"DD={self.dd_limit:.1%}, Corr={self.correlation_limit:.2f}, RiskThreshold={self.risk_score_critical:.2f}",
                context="risk_system_startup"
            )
        )
        
    def pre_trade_check(self, info_bus: InfoBus) -> Tuple[bool, str]:
        """🔧 FIXED: Enhanced pre-trade risk checks with correct scoring"""
        self.risk_stats['checks_performed'] += 1
        
        # Extract risk context from InfoBus
        risk_context = InfoBusExtractor.extract_risk_context(info_bus)
        
        # Check drawdown (percentage-based)
        current_dd = risk_context.get('drawdown_pct', 0.0) / 100.0  # Convert to decimal
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
        
        # 🔧 FIX: Check overall risk score (now properly 0-1 range)
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        
        # Validate risk score is in correct range
        if not (0.0 <= risk_score <= 1.0):
            self.logger.warning(
                format_operator_message(
                    "⚠️", "INVALID_RISK_SCORE",
                    details=f"Risk score {risk_score:.3f} outside valid range [0,1]",
                    result="Clamping to valid range",
                    context="risk_validation"
                )
            )
            risk_score = np.clip(risk_score, 0.0, 1.0)
        
        # Log risk score for debugging
        self.logger.debug(
            format_operator_message(
                "📊", "RISK_SCORE_CHECK",
                details=f"Current: {risk_score:.3f}, Threshold: {self.risk_score_critical:.3f}",
                context="risk_monitoring"
            )
        )
        
        # Check against threshold
        if risk_score > self.risk_score_critical:
            return self._handle_risk_violation(
                info_bus, "RISK_SCORE",
                f"Risk score {risk_score:.3f} exceeds threshold {self.risk_score_critical:.3f}"
            )
        
        # Warning level
        if risk_score > self.risk_score_warning:
            self.logger.warning(
                format_operator_message(
                    "⚠️", "RISK_SCORE_WARNING",
                    details=f"Risk score {risk_score:.3f} above warning level {self.risk_score_warning:.3f}",
                    context="risk_monitoring"
                )
            )
        
        # All checks passed
        self._update_info_bus_risk_status(info_bus, "PASSED", "All risk checks passed")
        return True, "All risk checks passed"
    
    def _handle_risk_violation(self, info_bus: InfoBus, violation_type: str, 
                              message: str) -> Tuple[bool, str]:
        """Handle risk violation with proper logging and alerts"""
        self.risk_stats['violations_detected'] += 1
        self.risk_stats['trades_blocked'] += 1
        
        current_step = info_bus.get('step_idx', 0)
        if current_step - self.last_alerts[violation_type] < self.alert_cooldown:
            return False, message  # Silent block
        
        self.last_alerts[violation_type] = current_step
        self.risk_stats['alerts_raised'] += 1
        
        self.logger.warning(
            format_operator_message(
                "🚫", "TRADE_BLOCKED",
                details=message,
                result="Trade execution prevented",
                context="risk_management"
            )
        )
        
        InfoBusUpdater.add_alert(info_bus, message, severity="warning", module="RiskManager")
        self._update_info_bus_risk_status(info_bus, "VIOLATION", message)
        
        self.audit_tracker.record_event(
            "risk_violation",
            "RiskManager", 
            {"type": violation_type, "message": message},
            severity="warning"
        )
        
        return False, message
    
    def post_trade_update(self, info_bus: InfoBus, trade: Dict[str, Any]):
        """Update risk systems after trade with InfoBus coordination"""
        pnl = trade.get('pnl', 0.0)
        instrument = trade.get('instrument', 'UNKNOWN')
        
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
                'positions': self.max_positions,
                'risk_score_warning': self.risk_score_warning,
                'risk_score_critical': self.risk_score_critical
            }
        }
        
        InfoBusUpdater.update_risk_snapshot(info_bus, risk_status)

    def get_risk_status_report(self) -> str:
        """Generate operator-friendly risk status report"""
        
        return f"""
🛡️ FIXED RISK MANAGER STATUS
═══════════════════════════════════════
📊 Risk Score Thresholds (FIXED 0-1 range):
• Warning Level: {self.risk_score_warning:.2f}
• Critical Level: {self.risk_score_critical:.2f}

⚖️ RISK LIMITS
• Max Drawdown: {self.dd_limit:.1%}
• Max Correlation: {self.correlation_limit:.2f}
• Max Positions: {self.max_positions}

📈 PERFORMANCE STATS
• Checks Performed: {self.risk_stats['checks_performed']}
• Violations Detected: {self.risk_stats['violations_detected']}
• Trades Blocked: {self.risk_stats['trades_blocked']}
• Alerts Raised: {self.risk_stats['alerts_raised']}

✅ STATUS: Risk scoring fixed to proper 0-1 range
        """


# ═══════════════════════════════════════════════════════════════════
# Enhanced Environment Utilities (keeping existing structure)
# ═══════════════════════════════════════════════════════════════════

def create_enhanced_trading_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create enhanced trading environment with fixed risk management"""
    
    # 🔧 FIX: Ensure risk config uses proper thresholds
    risk_config = config.get('risk_config', {})
    risk_config.update({
        'risk_score_warning': 0.7,    # Warning at 70% risk
        'risk_score_critical': 0.8,   # Block at 80% risk
        'dd_limit': 0.15,             # 15% max drawdown
        'correlation_limit': 0.85,    # 85% max correlation
        'max_positions': 8            # Max 8 positions
    })
    
    return {
        'risk_manager': UnifiedRiskManager(risk_config),
        'pipeline': None,  # To be initialized with modules
        'config': config,
        'status': 'initialized'
    }


def validate_risk_configuration(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate risk configuration has proper ranges"""
    
    issues = []
    
    risk_config = config.get('risk_config', {})
    
    # Check risk score thresholds are in 0-1 range
    warning_threshold = risk_config.get('risk_score_warning', 0.7)
    critical_threshold = risk_config.get('risk_score_critical', 0.8)
    
    if not (0.0 <= warning_threshold <= 1.0):
        issues.append(f"Risk warning threshold {warning_threshold} not in [0,1] range")
    
    if not (0.0 <= critical_threshold <= 1.0):
        issues.append(f"Risk critical threshold {critical_threshold} not in [0,1] range")
    
    if warning_threshold >= critical_threshold:
        issues.append(f"Warning threshold {warning_threshold} should be < critical {critical_threshold}")
    
    # Check drawdown limit
    dd_limit = risk_config.get('dd_limit', 0.15)
    if not (0.0 < dd_limit <= 1.0):
        issues.append(f"Drawdown limit {dd_limit} not in (0,1] range")
    
    return len(issues) == 0, issues


def get_system_integration_status() -> Dict[str, Any]:
    """Get comprehensive system integration status"""
    
    return {
        'infobus_integration': '✅ Complete',
        'risk_management': '🔧 FIXED - Proper 0-1 scoring',
        'market_data_access': '🔧 FIXED - Multiple extraction methods',
        'ml_model_dimensions': '🔧 FIXED - Consistent feature vectors',
        'quality_validation': '🔧 FIXED - Added missing attributes',
        'log_rotation': '✅ Implemented - 2000 lines max',
        'operator_logging': '✅ Human-readable messages',
        'audit_trails': '✅ Comprehensive tracking',
        'module_pipeline': '✅ Error-resilient processing',
        'backward_compatibility': '✅ Legacy interfaces maintained'
    }