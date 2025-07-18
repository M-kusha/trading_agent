"""
Enhanced Utilities for Modern SmartInfoBus Trading Environment
Zero-legacy, production-ready implementations
"""

import time
import logging
import threading
from functools import wraps
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class SystemHealth:
    """System health status"""
    smartinfobus_active: bool = False
    modules_active: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_space_gb: float = 0.0
    last_update: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # No need to initialize since we use field(default_factory=list)
        pass


def profile_method(func):
    """Performance profiling decorator for environment methods"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Log slow methods
            if hasattr(self, 'logger') and elapsed_ms > 100:
                self.logger.debug(f"⏱️ {func.__name__} took {elapsed_ms:.1f}ms")
            
            # Store performance metrics if SmartInfoBus available
            if hasattr(self, 'smart_bus') and self.smart_bus:
                self.smart_bus.set(
                    f'performance_{func.__name__}',
                    {'duration_ms': elapsed_ms, 'timestamp': time.time()},
                    module='Environment',
                    thesis=f"Performance metric for {func.__name__}"
                )
            
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            if hasattr(self, 'logger'):
                self.logger.error(f"❌ {func.__name__} failed after {elapsed_ms:.1f}ms: {e}")
            raise
    return wrapper


def safe_import(module_name: str, fallback=None):
    """Safely import modules with fallback"""
    try:
        if module_name == "modules.utils.info_bus":
            from modules.utils.info_bus import InfoBusManager
            return InfoBusManager
        elif module_name == "modules.utils.audit_utils":
            from modules.utils.audit_utils import RotatingLogger
            return RotatingLogger
        elif module_name == "modules.core.module_system":
            from modules.core.module_system import ModuleOrchestrator
            return ModuleOrchestrator
        else:
            __import__(module_name)
            return True
    except ImportError:
        return fallback


def create_enhanced_logger(name: str, log_path: Optional[str] = None) -> logging.Logger:
    """Create enhanced logger with proper formatting"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler if path provided
        if log_path:
            try:
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                file_handler = logging.FileHandler(log_path)
                file_formatter = logging.Formatter(
                    '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
                )
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                logger.warning(f"Could not create file handler: {e}")
        
        logger.setLevel(logging.INFO)
    
    return logger


def validate_trading_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Enhanced trading configuration validation"""
    warnings = []
    errors = []
    
    # Critical validations
    required_fields = ['initial_balance', 'max_steps', 'instruments', 'max_drawdown']
    for field in required_fields:
        if field not in config or config[field] is None:
            errors.append(f"❌ Missing required field: {field}")
    
    # Risk validations
    if config.get('max_drawdown', 0) > 0.5:
        warnings.append("⚠️ Max drawdown > 50% is extremely risky")
    elif config.get('max_drawdown', 0) > 0.3:
        warnings.append("⚠️ Max drawdown > 30% is very risky")
    
    if config.get('max_position_pct', 0) > 0.3:
        warnings.append("⚠️ Position size > 30% per trade is very risky")
    elif config.get('max_position_pct', 0) > 0.15:
        warnings.append("⚠️ Position size > 15% per trade is risky")
    
    if config.get('max_total_exposure', 0) > 1.0:
        errors.append("❌ Total exposure > 100% is invalid")
    elif config.get('max_total_exposure', 0) > 0.5:
        warnings.append("⚠️ Total exposure > 50% is very risky")
    
    # Live trading validations
    if config.get('live_mode', False):
        if not config.get('info_bus_enabled', True):
            warnings.append("⚠️ InfoBus strongly recommended for live trading")
        
        if config.get('debug', False) and config.get('max_position_pct', 0) > 0.05:
            warnings.append("⚠️ Large positions in live debug mode")
        
        if config.get('initial_balance', 0) < 1000:
            warnings.append("⚠️ Very small balance for live trading")
    
    # Performance validations
    if config.get('log_rotation_lines', 2000) > 10000:
        warnings.append("⚠️ Very high log rotation may impact performance")
    
    if config.get('max_steps', 200) > 1000:
        warnings.append("⚠️ Very long episodes may be slow")
    
    return len(errors) == 0, warnings + errors


def get_system_status() -> SystemHealth:
    """Get comprehensive system status"""
    health = SystemHealth()
    health.last_update = time.time()
    
    try:
        # Check SmartInfoBus
        InfoBusManager = safe_import("modules.utils.info_bus")
        if InfoBusManager and hasattr(InfoBusManager, 'get_instance'):
            try:
                # Type assertion to help type checker
                if hasattr(InfoBusManager, 'get_instance'):
                    smart_bus = InfoBusManager.get_instance()  # type: ignore
                health.smartinfobus_active = True
                health.modules_active = len(getattr(smart_bus, '_data_store', {}))
            except Exception as e:
                health.errors.append(f"SmartInfoBus error: {e}")
        
        # Check system resources
        try:
            import psutil
            process = psutil.Process()
            health.memory_usage_mb = process.memory_info().rss / 1024 / 1024
            health.cpu_usage_percent = process.cpu_percent()
            health.disk_space_gb = psutil.disk_usage('.').free / 1024 / 1024 / 1024
        except ImportError:
            health.warnings.append("psutil not available for system monitoring")
        except Exception as e:
            health.warnings.append(f"System monitoring error: {e}")
    
    except Exception as e:
        health.errors.append(f"System status check failed: {e}")
    
    return health


def create_fallback_systems():
    """Create fallback systems when SmartInfoBus is unavailable"""
    
    class FallbackSmartBus:
        """Minimal SmartInfoBus implementation"""
        def __init__(self):
            self._data_store = {}
            self._module_disabled = set()
        
        def set(self, key: str, value: Any, module: Optional[str] = None, thesis: Optional[str] = None):
            self._data_store[key] = {
                'value': value,
                'module': module,
                'thesis': thesis,
                'timestamp': time.time()
            }
        
        def get(self, key: str, module: Optional[str] = None):
            data = self._data_store.get(key)
            return data['value'] if data else None
        
        def register_provider(self, module: str, keys: List[str]):
            pass
        
        def register_consumer(self, module: str, keys: List[str]):
            pass
        
        def get_performance_metrics(self):
            return {
                'data_keys': len(self._data_store),
                'disabled_modules': len(self._module_disabled),
                'active': True
            }
    
    class FallbackOrchestrator:
        """Minimal orchestrator implementation"""
        def __init__(self):
            self.modules = []
            self.enabled = False
        
        def initialize(self):
            pass
        
        async def execute_step(self, inputs: Dict):
            return {}
    
    return FallbackSmartBus(), FallbackOrchestrator()


def validate_market_data(data_dict: Dict[str, Dict[str, pd.DataFrame]]) -> Tuple[bool, List[str]]:
    """Validate market data structure and quality"""
    issues = []
    
    if not data_dict:
        issues.append("❌ No market data provided")
        return False, issues
    
    required_columns = {'open', 'high', 'low', 'close'}
    
    for instrument, timeframes in data_dict.items():
        if not timeframes:
            issues.append(f"❌ No timeframes for {instrument}")
            continue
        
        for timeframe, df in timeframes.items():
            if df.empty:
                issues.append(f"⚠️ Empty data for {instrument}/{timeframe}")
                continue
            
            # Check required columns
            missing_cols = required_columns - set(df.columns)
            if missing_cols:
                issues.append(f"❌ Missing columns in {instrument}/{timeframe}: {missing_cols}")
            
            # Check data quality
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    if bool(df[col].isnull().any()):
                        null_count = df[col].isnull().sum()
                        issues.append(f"⚠️ {null_count} null values in {instrument}/{timeframe}.{col}")
                    
                    if (df[col] <= 0).any():
                        zero_count = (df[col] <= 0).sum()
                        issues.append(f"⚠️ {zero_count} non-positive values in {instrument}/{timeframe}.{col}")
            
            # Check OHLC logic
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                high_low_issues = (df['high'] < df['low']).sum()
                if high_low_issues > 0:
                    issues.append(f"❌ {high_low_issues} bars where high < low in {instrument}/{timeframe}")
                
                price_issues = (
                    (df['open'] > df['high']) | 
                    (df['open'] < df['low']) |
                    (df['close'] > df['high']) | 
                    (df['close'] < df['low'])
                ).sum()
                if price_issues > 0:
                    issues.append(f"❌ {price_issues} OHLC logic violations in {instrument}/{timeframe}")
    
    # Summary
    if not issues:
        issues.append("✅ Market data validation passed")
    
    has_critical_issues = any(issue.startswith("❌") for issue in issues)
    return not has_critical_issues, issues


def optimize_environment_performance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize environment configuration for performance"""
    optimized = config.copy()
    
    # Reduce logging overhead in production
    if not optimized.get('debug', False):
        optimized['log_rotation_lines'] = min(optimized.get('log_rotation_lines', 2000), 1000)
        optimized['info_bus_audit_level'] = 'WARNING'  # Less verbose
    
    # Optimize for Windows
    import platform
    if platform.system() == "Windows":
        optimized['num_envs'] = 1  # Single environment for stability
        optimized['enable_parallel_processing'] = False
    
    # Memory optimizations
    if optimized.get('max_steps', 200) > 500:
        optimized['max_history'] = 50  # Reduce memory footprint
    
    # Live trading optimizations
    if optimized.get('live_mode', False):
        optimized['risk_check_frequency'] = 1  # More frequent risk checks
        optimized['max_concurrent_alerts'] = 5  # Limit alert spam
        optimized['enable_shadow_sim'] = False  # Reduce overhead
    
    return optimized


def create_environment_diagnostics(env) -> Dict[str, Any]:
    """Create comprehensive environment diagnostics"""
    diagnostics = {
        'timestamp': time.time(),
        'environment_type': type(env).__name__,
        'configuration': {},
        'system_status': {},
        'performance_metrics': {},
        'health_status': 'unknown'
    }
    
    try:
        # Basic environment info
        if hasattr(env, 'config'):
            diagnostics['configuration'] = {
                'instruments': getattr(env.config, 'instruments', []),
                'max_steps': getattr(env.config, 'max_steps', None),
                'live_mode': getattr(env.config, 'live_mode', False),
                'info_bus_enabled': getattr(env.config, 'info_bus_enabled', False)
            }
        
        # SmartInfoBus status
        if hasattr(env, 'smart_bus') and env.smart_bus:
            try:
                diagnostics['system_status']['smartinfobus'] = env.smart_bus.get_performance_metrics()
            except:
                diagnostics['system_status']['smartinfobus'] = 'error'
        
        # Module system status
        if hasattr(env, 'orchestrator') and env.orchestrator:
            diagnostics['system_status']['modules'] = len(getattr(env.orchestrator, 'modules', []))
        
        # Environment-specific metrics
        if hasattr(env, 'market_state'):
            diagnostics['performance_metrics'] = {
                'current_step': getattr(env.market_state, 'current_step', 0),
                'balance': getattr(env.market_state, 'balance', 0),
                'drawdown': getattr(env.market_state, 'current_drawdown', 0),
                'episode_count': getattr(env, 'episode_count', 0)
            }
        
        # Overall health assessment
        has_smartinfobus = diagnostics['system_status'].get('smartinfobus') != 'error'
        has_modules = diagnostics['system_status'].get('modules', 0) > 0
        
        if has_smartinfobus and has_modules:
            diagnostics['health_status'] = 'excellent'
        elif has_smartinfobus:
            diagnostics['health_status'] = 'good'
        else:
            diagnostics['health_status'] = 'basic'
    
    except Exception as e:
        diagnostics['error'] = str(e)
        diagnostics['health_status'] = 'error'
    
    return diagnostics


# Export utilities
__all__ = [
    'SystemHealth',
    'profile_method',
    'safe_import',
    'create_enhanced_logger',
    'validate_trading_config',
    'get_system_status',
    'create_fallback_systems',
    'validate_market_data',
    'optimize_environment_performance',
    'create_environment_diagnostics'
]