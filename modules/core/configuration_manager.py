#!/usr/bin/env python3
"""
Configuration Manager for SmartInfoBus System
Centralized configuration loading and distribution to modules
"""

import os
import yaml
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict
import logging

from modules.utils.audit_utils import RotatingLogger, format_operator_message


@dataclass
class ModuleConfigSpec:
    """Configuration specification for a module"""
    name: str
    category: str
    config_section: str  # Which section in YAML contains this module's config
    default_config: Dict[str, Any] = field(default_factory=dict)
    required_keys: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Callable] = field(default_factory=dict)


class ConfigurationManager:
    """
    Production-grade configuration manager for SmartInfoBus system.
    Loads configurations from YAML files and distributes to modules.
    """
    
    _instance: Optional['ConfigurationManager'] = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize configuration manager"""
        
        # Configuration file paths
        self.config_paths = {
            'system': Path('config/system_config.yaml'),
            'risk': Path('config/risk_policy.yaml'),
            'explainability': Path('config/explainability_standards.yaml')
        }
        
        # Loaded configurations
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.module_configs: Dict[str, Dict[str, Any]] = {}
        
        # Module specifications
        self.module_specs: Dict[str, ModuleConfigSpec] = {}
        
        # Configuration watchers
        self.config_watchers: List[Callable] = []
        self.module_watchers: Dict[str, List[Callable]] = defaultdict(list)
        
        # File monitoring
        self.file_timestamps: Dict[str, float] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Setup logging
        self.logger = RotatingLogger(
            name="ConfigurationManager",
            log_path="logs/config/configuration_manager.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Initialize
        self._initialize_module_specs()
        self._load_all_configurations()
        self._start_monitoring()
        
        self.logger.info(
            format_operator_message(
                "⚙️", "CONFIGURATION MANAGER INITIALIZED",
                details=f"Loaded {len(self.configs)} config files",
                context="startup"
            )
        )
    
    @classmethod
    def get_instance(cls) -> 'ConfigurationManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def _initialize_module_specs(self):
        """Initialize module configuration specifications"""
        
        # Market Analysis Modules
        self.module_specs['MarketThemeDetector'] = ModuleConfigSpec(
            name='MarketThemeDetector',
            category='market',
            config_section='modules.MarketThemeDetector.config',
            default_config={
                'lookback_periods': [5, 10, 20, 50],
                'confidence_threshold': 0.7,
                'theme_categories': ['bullish', 'bearish', 'ranging', 'breakout'],
                'timeout_ms': 150
            },
            required_keys=['lookback_periods', 'confidence_threshold'],
            validation_rules={
                'confidence_threshold': lambda x: 0 <= x <= 1,
                'lookback_periods': lambda x: isinstance(x, list) and len(x) > 0
            }
        )
        
        self.module_specs['AdvancedFeatureEngine'] = ModuleConfigSpec(
            name='AdvancedFeatureEngine',
            category='market',
            config_section='modules.AdvancedFeatureEngine.config',
            default_config={
                'feature_sets': ['technical', 'statistical', 'momentum', 'volatility'],
                'normalization': 'z_score',
                'feature_selection': True,
                'timeout_ms': 100
            },
            required_keys=['feature_sets'],
            validation_rules={
                'feature_sets': lambda x: isinstance(x, list) and len(x) > 0
            }
        )
        
        # Strategy Modules
        self.module_specs['StrategyGenomePool'] = ModuleConfigSpec(
            name='StrategyGenomePool',
            category='strategy',
            config_section='modules.StrategyGenomePool.config',
            default_config={
                'population_size': 50,
                'mutation_rate': 0.1,
                'selection_pressure': 0.7,
                'genome_types': ['momentum', 'mean_reversion', 'breakout', 'arbitrage'],
                'timeout_ms': 200
            },
            required_keys=['population_size', 'mutation_rate'],
            validation_rules={
                'population_size': lambda x: x > 0,
                'mutation_rate': lambda x: 0 <= x <= 1
            }
        )
        
        # Risk Management Modules
        self.module_specs['RiskManager'] = ModuleConfigSpec(
            name='RiskManager',
            category='risk',
            config_section='portfolio_limits',
            default_config={
                'max_portfolio_var_daily': 0.02,
                'max_drawdown_limit': 0.15,
                'max_position_size_pct': 0.10,
                'timeout_ms': 50
            },
            required_keys=['max_portfolio_var_daily', 'max_drawdown_limit'],
            validation_rules={
                'max_portfolio_var_daily': lambda x: 0 < x < 1,
                'max_drawdown_limit': lambda x: 0 < x < 1
            }
        )
        
        # Meta Learning Modules
        self.module_specs['PPOAgent'] = ModuleConfigSpec(
            name='PPOAgent',
            category='meta',
            config_section='modules.MetaRLController.config',
            default_config={
                'learning_rate': 0.0003,
                'clip_eps': 0.2,
                'value_coeff': 0.5,
                'entropy_coeff': 0.01,
                'timeout_ms': 300
            },
            required_keys=['learning_rate'],
            validation_rules={
                'learning_rate': lambda x: x > 0
            }
        )
        
        # Add more module specs as needed...
        
        self.logger.info(f"Initialized {len(self.module_specs)} module specifications")
    
    def _load_all_configurations(self):
        """Load all configuration files"""
        for config_name, config_path in self.config_paths.items():
            self._load_configuration_file(config_name, config_path)
        
        # Build module-specific configurations
        self._build_module_configurations()
    
    def _load_configuration_file(self, name: str, path: Path):
        """Load a single configuration file"""
        try:
            if not path.exists():
                self.logger.warning(f"Configuration file not found: {path}")
                self.configs[name] = {}
                return
            
            # Try different encodings to handle encoding issues
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            config_data = None
            
            for encoding in encodings_to_try:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        config_data = yaml.safe_load(f)
                    self.logger.debug(f"Successfully loaded {name} with {encoding} encoding")
                    break
                except UnicodeDecodeError as e:
                    self.logger.debug(f"Failed to load {name} with {encoding}: {e}")
                    continue
                except Exception as e:
                    self.logger.debug(f"Failed to load {name} with {encoding}: {e}")
                    continue
            
            if config_data is None:
                self.logger.error(f"Failed to load configuration {name} with any encoding")
                self.configs[name] = {}
                return
            
            self.configs[name] = config_data or {}
            self.file_timestamps[name] = path.stat().st_mtime
            
            self.logger.info(f"Loaded configuration: {name} from {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration {name}: {e}")
            self.configs[name] = {}
    
    def _build_module_configurations(self):
        """Build module-specific configurations from loaded YAML files"""
        for module_name, spec in self.module_specs.items():
            try:
                module_config = self._extract_module_config(spec)
                self.module_configs[module_name] = module_config
                
                self.logger.debug(f"Built configuration for {module_name}: {len(module_config)} keys")
                
            except Exception as e:
                self.logger.error(f"Failed to build config for {module_name}: {e}")
                self.module_configs[module_name] = spec.default_config.copy()
    
    def _extract_module_config(self, spec: ModuleConfigSpec) -> Dict[str, Any]:
        """Extract configuration for a specific module"""
        config = spec.default_config.copy()
        
        # Extract from system config
        if 'system' in self.configs:
            system_config = self.configs['system']
            
            # Navigate to the module's config section
            config_path = spec.config_section.split('.')
            current_config = system_config
            
            for path_part in config_path:
                if isinstance(current_config, dict) and path_part in current_config:
                    current_config = current_config[path_part]
                else:
                    current_config = None
                    break
            
            if current_config and isinstance(current_config, dict):
                config.update(current_config)
        
        # Extract from risk policy if applicable
        if spec.category == 'risk' and 'risk' in self.configs:
            risk_config = self.configs['risk']
            
            # Map risk policy sections to module config
            if spec.config_section in risk_config:
                risk_section = risk_config[spec.config_section]
                if isinstance(risk_section, dict):
                    config.update(risk_section)
        
        # Extract from explainability standards if applicable
        if 'explainability' in self.configs:
            explainability_config = self.configs['explainability']
            
            # Apply explainability standards
            if 'module_standards' in explainability_config:
                module_standards = explainability_config['module_standards']
                if spec.category in module_standards:
                    category_standards = module_standards[spec.category]
                    if isinstance(category_standards, dict):
                        # Add explainability requirements
                        config.update({
                            'explainability_level': category_standards.get('explainability_level', 'medium'),
                            'thesis_mandatory': category_standards.get('thesis_mandatory', True),
                            'explanation_depth': category_standards.get('explanation_depth', 'standard')
                        })
        
        # Validate configuration
        self._validate_module_config(spec, config)
        
        return config
    
    def _validate_module_config(self, spec: ModuleConfigSpec, config: Dict[str, Any]):
        """Validate module configuration"""
        # Check required keys
        for required_key in spec.required_keys:
            if required_key not in config:
                raise ValueError(f"Missing required key '{required_key}' for {spec.name}")
        
        # Apply validation rules
        for key, validator in spec.validation_rules.items():
            if key in config:
                if not validator(config[key]):
                    raise ValueError(f"Validation failed for {spec.name}.{key}: {config[key]}")
    
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module"""
        if module_name in self.module_configs:
            return self.module_configs[module_name].copy()
        
        # Return default config if module not found
        self.logger.warning(f"No configuration found for module: {module_name}")
        return {}
    
    def get_system_config(self) -> Dict[str, Any]:
        """Get system-wide configuration"""
        return self.configs.get('system', {}).copy()
    
    def get_risk_policy(self) -> Dict[str, Any]:
        """Get risk policy configuration"""
        return self.configs.get('risk', {}).copy()
    
    def get_explainability_standards(self) -> Dict[str, Any]:
        """Get explainability standards"""
        return self.configs.get('explainability', {}).copy()
    
    def register_module_spec(self, spec: ModuleConfigSpec):
        """Register a new module specification"""
        self.module_specs[spec.name] = spec
        
        # Build configuration for the new module
        try:
            module_config = self._extract_module_config(spec)
            self.module_configs[spec.name] = module_config
            
            self.logger.info(f"Registered configuration for module: {spec.name}")
            
        except Exception as e:
            self.logger.error(f"Failed to register config for {spec.name}: {e}")
            self.module_configs[spec.name] = spec.default_config.copy()
    
    def add_config_watcher(self, callback: Callable):
        """Add a configuration change watcher"""
        self.config_watchers.append(callback)
    
    def add_module_watcher(self, module_name: str, callback: Callable):
        """Add a module-specific configuration watcher"""
        self.module_watchers[module_name].append(callback)
    
    def _start_monitoring(self):
        """Start configuration file monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_files, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Started configuration file monitoring")
    
    def _monitor_files(self):
        """Monitor configuration files for changes"""
        while self.monitoring_active:
            try:
                for config_name, config_path in self.config_paths.items():
                    if config_path.exists():
                        current_mtime = config_path.stat().st_mtime
                        last_mtime = self.file_timestamps.get(config_name, 0)
                        
                        if current_mtime > last_mtime:
                            self.logger.info(f"Configuration file changed: {config_path}")
                            self._reload_configuration(config_name, config_path)
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Error monitoring configuration files: {e}")
                time.sleep(30)  # Back off on error
    
    def _reload_configuration(self, config_name: str, config_path: Path):
        """Reload a configuration file"""
        try:
            # Load the updated configuration
            old_config = self.configs.get(config_name, {}).copy()
            self._load_configuration_file(config_name, config_path)
            
            # Only rebuild if we actually loaded new data
            if self.configs[config_name]:
                # Rebuild module configurations
                old_module_configs = self.module_configs.copy()
                self._build_module_configurations()
                
                # Notify watchers
                for callback in self.config_watchers:
                    try:
                        callback(config_name, old_config, self.configs[config_name])
                    except Exception as e:
                        self.logger.error(f"Error in config watcher: {e}")
                
                # Notify module-specific watchers
                for module_name, callbacks in self.module_watchers.items():
                    if module_name in old_module_configs and module_name in self.module_configs:
                        old_module_config = old_module_configs[module_name]
                        new_module_config = self.module_configs[module_name]
                        
                        if old_module_config != new_module_config:
                            for callback in callbacks:
                                try:
                                    callback(module_name, old_module_config, new_module_config)
                                except Exception as e:
                                    self.logger.error(f"Error in module watcher for {module_name}: {e}")
                
                self.logger.info(f"Successfully reloaded configuration: {config_name}")
            else:
                self.logger.warning(f"Failed to load configuration {config_name} - keeping old config")
            
        except Exception as e:
            self.logger.error(f"Failed to reload configuration {config_name}: {e}")
    
    def stop_monitoring(self):
        """Stop configuration file monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        
        self.logger.info("Stopped configuration file monitoring")
    
    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration for ModuleOrchestrator"""
        system_config = self.get_system_config()
        return system_config.get('execution', {})
    
    def get_module_registry(self) -> Dict[str, Any]:
        """Get module registry configuration"""
        system_config = self.get_system_config()
        return system_config.get('modules', {})
    
    def get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence configuration"""
        system_config = self.get_system_config()
        return system_config.get('persistence', {})
    
    def get_hot_reload_config(self) -> Dict[str, Any]:
        """Get hot reload configuration"""
        system_config = self.get_system_config()
        return system_config.get('hot_reload', {})
    
    def __del__(self):
        """Cleanup on destruction"""
        self.stop_monitoring() 