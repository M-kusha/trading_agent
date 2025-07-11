"""
Shared utilities for Modern SmartInfoBus Trading Environment
Simplified for zero-wiring architecture
"""
from functools import wraps
import time
from typing import Dict, Any, Optional
import logging


def profile_method(func):
    """Simple method profiling decorator"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        try:
            result = func(self, *args, **kwargs)
            elapsed = (time.time() - start_time) * 1000
            
            if hasattr(self, 'logger') and elapsed > 100:  # Log slow methods
                self.logger.debug(f"{func.__name__} took {elapsed:.1f}ms")
            
            return result
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            if hasattr(self, 'logger'):
                self.logger.error(f"{func.__name__} failed after {elapsed:.1f}ms: {e}")
            raise
    return wrapper


def create_module_config(base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized module configuration"""
    return {
        'debug': base_config.get('debug', False),
        'max_history': base_config.get('max_history', 100),
        'audit_enabled': base_config.get('audit_enabled', True),
        'log_rotation_lines': base_config.get('log_rotation_lines', 2000),
        'health_check_interval': base_config.get('health_check_interval', 100),
        'info_bus_enabled': base_config.get('info_bus_enabled', True),
    }


def validate_trading_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate trading configuration"""
    warnings = []
    
    # Risk validation
    if config.get('max_drawdown', 0) > 0.5:
        warnings.append("⚠️ Max drawdown > 50% is extremely risky")
    
    if config.get('max_position_pct', 0) > 0.3:
        warnings.append("⚠️ Position size > 30% per trade is very risky")
    
    # SmartInfoBus validation
    if not config.get('info_bus_enabled', True):
        warnings.append("⚠️ SmartInfoBus disabled - functionality will be limited")
    
    return len(warnings) == 0, warnings


def get_system_status() -> Dict[str, Any]:
    """Get basic system status"""
    from modules.utils.info_bus import InfoBusManager
    
    try:
        smart_bus = InfoBusManager.get_instance()
        return {
            'smart_bus_active': True,
            'data_keys': len(smart_bus._data_store),
            'disabled_modules': list(smart_bus._module_disabled),
            'performance_ok': True
        }
    except Exception as e:
        return {
            'smart_bus_active': False,
            'error': str(e),
            'performance_ok': False
        }


class SimpleLogger:
    """Simple logging utility for environments that don't have logger setup"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, msg: str):
        self.logger.info(msg)
    
    def error(self, msg: str):
        self.logger.error(msg)
    
    def warning(self, msg: str):
        self.logger.warning(msg)
    
    def debug(self, msg: str):
        self.logger.debug(msg)