#!/usr/bin/env python3
"""
Configuration Manager for SmartInfoBus System
Centralized configuration loading and distribution to modules
"""

from __future__ import annotations

import os
import io
import yaml
import time
import json
import glob
import copy
import queue
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from modules.utils.audit_utils import RotatingLogger, format_operator_message

# ─────────────────────────────────────────────────────────────
# YAML helpers: deep-merge, env interpolation, custom tags
# ─────────────────────────────────────────────────────────────

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Non-destructive deep merge: return new dict of a←b (b overrides a)."""
    if not isinstance(a, dict) or not isinstance(b, dict):
        return copy.deepcopy(b)
    out = copy.deepcopy(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out

def _interpolate_env(value: Any) -> Any:
    """Interpolates ${ENV[:default]} tokens in strings recursively."""
    if isinstance(value, str):
        out = []
        i = 0
        s = value
        while i < len(s):
            if s[i:i+2] == "${":
                j = s.find("}", i+2)
                if j == -1:
                    out.append(s[i:])
                    break
                token = s[i+2:j]
                if ":" in token:
                    var, default = token.split(":", 1)
                else:
                    var, default = token, ""
                out.append(os.getenv(var, default))
                i = j + 1
            else:
                out.append(s[i])
                i += 1
        return "".join(out)
    elif isinstance(value, list):
        return [_interpolate_env(x) for x in value]
    elif isinstance(value, dict):
        return {k: _interpolate_env(v) for k, v in value.items()}
    return value

class _EnvLoader(yaml.SafeLoader):
    pass

def _yaml_include(loader: _EnvLoader, node):
    """!include path/to/file.yaml (relative to the including file)"""
    base = Path(getattr(loader, 'name', '.') or '.').parent
    path = Path(loader.construct_scalar(node))
    target = path if path.is_absolute() else (base / path)
    with open(target, 'r', encoding='utf-8') as f:
        data = yaml.load(f, _EnvLoader)
        return data

def _yaml_env(loader: _EnvLoader, node):
    """!env VAR[:default] — returns environment variable or default"""
    token = loader.construct_scalar(node)
    if ':' in token:
        var, default = token.split(':', 1)
    else:
        var, default = token, ''
    return os.getenv(var, default)

_EnvLoader.add_constructor('!include', _yaml_include)
_EnvLoader.add_constructor('!env', _yaml_env)

# ─────────────────────────────────────────────────────────────

@dataclass
class ModuleConfigSpec:
    """Configuration specification for a module"""
    name: str
    category: str
    config_section: str  # dot-path inside system config
    default_config: Dict[str, Any] = field(default_factory=dict)
    required_keys: List[str] = field(default_factory=list)
    # validator can be Callable[[Any], bool] OR Callable[[Any, Dict[str, Any]], bool]
    validation_rules: Dict[str, Callable] = field(default_factory=dict)
    # optional post-processor: Callable[[Dict[str, Any]], Dict[str, Any]]
    postprocess: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    # optional env prefix for overrides (e.g., SIB_PPOAGENT__LEARNING_RATE)
    env_prefix: Optional[str] = None
    # versioning (for migrations)
    schema_version: str = "1.0"

class ConfigurationManager:
    """
    Production-grade configuration manager for SmartInfoBus system.
    Loads configurations from YAML files and distributes to modules.
    """

    _instance: Optional['ConfigurationManager'] = None
    _lock = threading.Lock()

    # keys that look like secrets and should be redacted when logging
    _SECRET_KEYS = {'password', 'passwd', 'token', 'api_key', 'secret', 'bearer', 'client_secret'}

    def __init__(self):
        """Initialize configuration manager"""

        # Configuration file paths (primary + overlays)
        self.config_paths = {
            'system': Path('config/system_config.yaml'),
            'system_local': Path('config/system_config.local.yaml'),
            'system_dir': Path('config/system_config.d'),
            'risk': Path('config/risk_policy.yaml'),
            'explainability': Path('config/explainability_standards.yaml')
        }

        # Loaded configurations
        self.configs: Dict[str, Dict[str, Any]] = {}
        self.module_configs: Dict[str, Dict[str, Any]] = {}

        # Module specifications
        self.module_specs: Dict[str, ModuleConfigSpec] = {}

        # Configuration watchers
        self.config_watchers: List[Callable[[str, Dict[str, Any], Dict[str, Any]], None]] = []
        self.module_watchers: Dict[str, List[Callable[[str, Dict[str, Any], Dict[str, Any]], None]]] = defaultdict(list)

        # File monitoring
        self.file_timestamps: Dict[str, float] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._reload_events: "queue.Queue[Tuple[str, Path]]" = queue.Queue()
        self._debounce_ms = 400  # collapse bursty writes

        # Snapshots + hashing
        self._last_snapshot_hash: Dict[str, str] = {}
        self._module_runtime_overrides: Dict[str, Dict[str, Any]] = {}

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

    # ─────────────────────────────────────────────────────────
    # Singleton
    # ─────────────────────────────────────────────────────────
    @classmethod
    def get_instance(cls) -> 'ConfigurationManager':
        """Get singleton instance"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ─────────────────────────────────────────────────────────
    # Specs
    # ─────────────────────────────────────────────────────────
    def _initialize_module_specs(self):
        """Initialize module configuration specifications (defaults)."""

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

        self.logger.info(f"Initialized {len(self.module_specs)} module specifications")

    def register_module_spec(self, spec: ModuleConfigSpec):
        """Register or update a module specification and build its config."""
        self.module_specs[spec.name] = spec
        try:
            module_config = self._extract_module_config(spec)
            self.module_configs[spec.name] = module_config
            self.logger.info(f"Registered configuration for module: {spec.name}")
        except Exception as e:
            self.logger.error(f"Failed to register config for {spec.name}: {e}")
            self.module_configs[spec.name] = spec.default_config.copy()

    def register_module_specs(self, specs: List[ModuleConfigSpec]):
        """Bulk register multiple specs."""
        for s in specs:
            self.register_module_spec(s)

    # ─────────────────────────────────────────────────────────
    # Loading and building
    # ─────────────────────────────────────────────────────────
    def _load_all_configurations(self):
        """Load all configuration files with overlays and env interpolation."""
        # Base files
        self._load_configuration_file('system', self.config_paths['system'])
        # Overlays: config/system_config.local.yaml (optional)
        self._load_configuration_file('system_local', self.config_paths['system_local'], optional=True)
        # Overlays: config/system_config.d/*.yaml
        merged_dir = {}
        if self.config_paths['system_dir'].exists():
            for extra in sorted(glob.glob(str(self.config_paths['system_dir'] / "*.yaml"))):
                name = f"system_dir::{Path(extra).name}"
                self._load_configuration_file(name, Path(extra), optional=True)
                merged_dir = _deep_merge(merged_dir, self.configs.get(name, {}))

        # Merge overlay precedence: base <- dir/*.yaml <- local
        sys_base = self.configs.get('system', {})
        sys_dir = merged_dir
        sys_local = self.configs.get('system_local', {})
        system_final = _deep_merge(_deep_merge(sys_base, sys_dir), sys_local)

        # Risk + explainability (simple load)
        self._load_configuration_file('risk', self.config_paths['risk'], optional=True)
        self._load_configuration_file('explainability', self.config_paths['explainability'], optional=True)

        # Interpolate env & store
        system_final = _interpolate_env(system_final)
        self.configs['system'] = system_final

        # Build module-specific configurations
        self._build_module_configurations()

        # Record snapshot hash
        self._hash_and_snapshot('system', system_final)
        if 'risk' in self.configs:
            self._hash_and_snapshot('risk', self.configs['risk'])
        if 'explainability' in self.configs:
            self._hash_and_snapshot('explainability', self.configs['explainability'])

    def _hash_and_snapshot(self, name: str, payload: Dict[str, Any]):
        try:
            data = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
            h = hashlib.sha256(data).hexdigest()
            self._last_snapshot_hash[name] = h
            # Write lightweight snapshot occasionally
            snap_dir = Path("logs/config/snapshots"); snap_dir.mkdir(parents=True, exist_ok=True)
            with open(snap_dir / f"{name}.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, indent=2, ensure_ascii=False))
        except Exception as e:
            self.logger.warning(f"Snapshot failed for {name}: {e}")

    def _load_yaml_any(self, path: Path) -> Dict[str, Any]:
        """Load YAML with custom loader + env interpolation."""
        # Try a few encodings; read entire file atomically
        encodings = ('utf-8', 'utf-8-sig', 'latin-1', 'cp1252')
        last_exc = None
        for enc in encodings:
            try:
                with open(path, 'r', encoding=enc) as f:
                    text = f.read()
                stream = io.StringIO(text)
                setattr(_EnvLoader, 'name', str(path))
                data = yaml.load(stream, Loader=_EnvLoader)
                # Guarantee a mapping return shape; discard non-dict YAML roots
                return data if isinstance(data, dict) else {}
            except UnicodeDecodeError as e:
                last_exc = e
                continue
            except Exception as e:
                last_exc = e
                continue
        raise last_exc or RuntimeError("Unknown YAML error")

    def _load_configuration_file(self, name: str, path: Path, optional: bool = False):
        """Load a single configuration file."""
        try:
            if not path.exists():
                if not optional:
                    self.logger.warning(f"Configuration file not found: {path}")
                self.configs[name] = {}
                return
            cfg = self._load_yaml_any(path)
            cfg = _interpolate_env(cfg)
            self.configs[name] = cfg
            self.file_timestamps[name] = path.stat().st_mtime
            self.logger.info(f"Loaded configuration: {name} from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration {name}: {e}")
            self.configs[name] = {}

    def _build_module_configurations(self):
        """Build module-specific configurations from loaded YAML files + runtime overrides."""
        built: Dict[str, Dict[str, Any]] = {}
        for module_name, spec in self.module_specs.items():
            try:
                module_config = self._extract_module_config(spec)
                # Runtime overrides (e.g., tests or interactive changes)
                if module_name in self._module_runtime_overrides:
                    module_config = _deep_merge(module_config, self._module_runtime_overrides[module_name])
                built[module_name] = module_config
                self.logger.debug(f"Built configuration for {module_name}: {len(module_config)} keys")
            except Exception as e:
                self.logger.error(f"Failed to build config for {module_name}: {e}")
                built[module_name] = spec.default_config.copy()
        # Diff & notify watchers
        old = self.module_configs
        self.module_configs = built
        for module_name, callbacks in self.module_watchers.items():
            old_cfg = old.get(module_name, {})
            new_cfg = built.get(module_name, {})
            if old_cfg != new_cfg:
                for cb in callbacks:
                    try:
                        cb(module_name, self._redacted(old_cfg), self._redacted(new_cfg))
                    except Exception as e:
                        self.logger.error(f"Error in module watcher for {module_name}: {e}")

    def _extract_module_config(self, spec: ModuleConfigSpec) -> Dict[str, Any]:
        """Extract configuration for a specific module."""
        config = copy.deepcopy(spec.default_config)

        # 1) from system config (dot-path)
        sys_cfg = self.configs.get('system', {})
        current = sys_cfg
        for part in spec.config_section.split('.'):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                current = None
                break
        if current and isinstance(current, dict):
            config = _deep_merge(config, current)

        # 2) category-level standards from explainability
        expl = self.configs.get('explainability', {})
        if isinstance(expl, dict):
            module_standards = expl.get('module_standards', {})
            if isinstance(module_standards, dict):
                cat_std = module_standards.get(spec.category, {})
                if isinstance(cat_std, dict):
                    explain_bits = {
                        'explainability_level': cat_std.get('explainability_level', 'medium'),
                        'thesis_mandatory': cat_std.get('thesis_mandatory', True),
                        'explanation_depth': cat_std.get('explanation_depth', 'standard')
                    }
                    config = _deep_merge(config, explain_bits)

        # 3) risk overlays if module is risk
        if spec.category == 'risk':
            risk_cfg = self.configs.get('risk', {})
            # support both dot-path key and flat top-level mapping
            if isinstance(risk_cfg, dict):
                if spec.config_section in risk_cfg and isinstance(risk_cfg[spec.config_section], dict):
                    config = _deep_merge(config, risk_cfg[spec.config_section])

        # 4) Env overrides by prefix (e.g., SIB_PPOAGENT__LEARNING_RATE=...)
        if spec.env_prefix:
            prefix = spec.env_prefix.upper().rstrip('_') + '__'
            for k, v in list(os.environ.items()):
                if k.startswith(prefix):
                    key = k[len(prefix):].lower()
                    try:
                        # attempt to parse JSON for complex values; fallback to raw string
                        parsed = json.loads(v)
                    except Exception:
                        parsed = v
                    config[key] = parsed

        # Validate
        self._validate_module_config(spec, config)

        # Postprocess
        if spec.postprocess:
            try:
                config = spec.postprocess(copy.deepcopy(config))
            except Exception as e:
                self.logger.error(f"Postprocess failed for {spec.name}: {e}")

        return config

    def _validate_module_config(self, spec: ModuleConfigSpec, config: Dict[str, Any]):
        """Validate module configuration."""
        # Schema/version check (optional)
        cfg_version = str(config.get('_schema_version', spec.schema_version))
        if cfg_version != spec.schema_version:
            self.logger.warning(f"{spec.name}: schema version mismatch (cfg={cfg_version}, expected={spec.schema_version})")

        # Required keys
        for required_key in spec.required_keys:
            if required_key not in config:
                raise ValueError(f"Missing required key '{required_key}' for {spec.name}")

        # Validation rules
        for key, validator in spec.validation_rules.items():
            if key in config:
                try:
                    ok = validator(config[key], config) if validator.__code__.co_argcount >= 2 else validator(config[key])
                except Exception as e:
                    raise ValueError(f"Validator error for {spec.name}.{key}: {e}")
                if not ok:
                    raise ValueError(f"Validation failed for {spec.name}.{key}: {config[key]}")

    # ─────────────────────────────────────────────────────────
    # Public getters
    # ─────────────────────────────────────────────────────────
    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get configuration for a specific module (copy)."""
        if module_name in self.module_configs:
            return copy.deepcopy(self.module_configs[module_name])
        self.logger.warning(f"No configuration found for module: {module_name}")
        return {}

    def get_system_config(self) -> Dict[str, Any]:
        """Get system-wide configuration."""
        return copy.deepcopy(self.configs.get('system', {}))

    def get_risk_policy(self) -> Dict[str, Any]:
        """Get risk policy configuration."""
        return copy.deepcopy(self.configs.get('risk', {}))

    def get_explainability_standards(self) -> Dict[str, Any]:
        """Get explainability standards."""
        return copy.deepcopy(self.configs.get('explainability', {}))

    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration for ModuleOrchestrator."""
        return self.get_system_config().get('execution', {})

    def get_module_registry(self) -> Dict[str, Any]:
        """Get module registry configuration."""
        return self.get_system_config().get('modules', {})

    def get_persistence_config(self) -> Dict[str, Any]:
        """Get persistence configuration."""
        return self.get_system_config().get('persistence', {})

    def get_hot_reload_config(self) -> Dict[str, Any]:
        """Get hot reload configuration."""
        return self.get_system_config().get('hot_reload', {})

    def get_monitoring_config(self) -> Dict[str, Any]:
        """Observability/monitoring configuration (for MonitoringHub)."""
        default = {'publish_interval_s': 60}
        mon = self.get_system_config().get('monitoring', {})
        return {**default, **mon} if isinstance(mon, dict) else default

    # ─────────────────────────────────────────────────────────
    # Runtime overrides (tests/ops) and watchers
    # ─────────────────────────────────────────────────────────
    def apply_runtime_overrides(self, module_name: str, overrides: Dict[str, Any]):
        """Apply in-memory overrides for a module (not persisted)."""
        if not isinstance(overrides, dict):
            raise TypeError("overrides must be a dict")
        self._module_runtime_overrides[module_name] = _deep_merge(
            self._module_runtime_overrides.get(module_name, {}), overrides
        )
        self._build_module_configurations()

    def clear_runtime_overrides(self, module_name: Optional[str] = None):
        if module_name is None:
            self._module_runtime_overrides.clear()
        else:
            self._module_runtime_overrides.pop(module_name, None)
        self._build_module_configurations()

    def add_config_watcher(self, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        """Add a configuration change watcher: (name, old, new)."""
        self.config_watchers.append(callback)

    def add_module_watcher(self, module_name: str, callback: Callable[[str, Dict[str, Any], Dict[str, Any]], None]):
        """Add a module-specific configuration watcher: (module_name, old, new)."""
        self.module_watchers[module_name].append(callback)

    # ─────────────────────────────────────────────────────────
    # Monitoring / reloading
    # ─────────────────────────────────────────────────────────
    def _start_monitoring(self):
        """Start configuration file monitoring."""
        if self.monitoring_active:
            return
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("Started configuration file monitoring")

    def stop_monitoring(self):
        """Stop configuration file monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
        self.logger.info("Stopped configuration file monitoring")

    def _monitor_loop(self):
        """Coalescing file watcher loop (polling)."""
        # Initial seeds
        to_watch: List[Tuple[str, Path]] = [
            ('system', self.config_paths['system']),
            ('system_local', self.config_paths['system_local']),
            ('risk', self.config_paths['risk']),
            ('explainability', self.config_paths['explainability']),
        ]
        # dynamic .d directory files
        def refresh_dir_files():
            if self.config_paths['system_dir'].exists():
                for extra in glob.glob(str(self.config_paths['system_dir'] / "*.yaml")):
                    name = f"system_dir::{Path(extra).name}"
                    to_watch.append((name, Path(extra)))

        refresh_dir_files()

        last_check = 0.0
        while self.monitoring_active:
            try:
                now = time.time()
                if now - last_check >= 1.0:
                    last_check = now
                    # scan
                    for name, p in to_watch:
                        try:
                            if p.exists():
                                mtime = p.stat().st_mtime
                                prev = self.file_timestamps.get(name, 0)
                                if mtime > prev:
                                    self._reload_events.put((name, p))
                                    self.file_timestamps[name] = mtime
                            else:
                                # file removed — treat as empty reload
                                if name in self.file_timestamps:
                                    self._reload_events.put((name, p))
                                    self.file_timestamps[name] = 0
                        except Exception:
                            continue

                    # refresh dir list periodically
                    if int(now) % 10 == 0:
                        to_watch = [t for t in to_watch if not t[0].startswith('system_dir::')]
                        refresh_dir_files()

                # Debounce + handle events
                try:
                    name, path = self._reload_events.get(timeout=0.25)
                    # coalesce burst
                    time.sleep(self._debounce_ms / 1000.0)
                    pending = [(name, path)]
                    while not self._reload_events.empty():
                        pending.append(self._reload_events.get_nowait())

                    # actually reload
                    changed_names = {n for n, _ in pending}
                    self._reload_changed_files(changed_names)
                except queue.Empty:
                    pass

            except Exception as e:
                self.logger.error(f"Error monitoring configuration files: {e}")
                time.sleep(2.0)

    def _reload_changed_files(self, names: set):
        """Reload a set of changed config files and notify watchers."""
        old_all = {k: copy.deepcopy(v) for k, v in self.configs.items()}

        # Reload targeted names
        for name in names:
            # Map back to canonical slots (system, system_local, system_dir::* etc.)
            if name == 'system':
                self._load_configuration_file('system', self.config_paths['system'], optional=True)
            elif name == 'system_local':
                self._load_configuration_file('system_local', self.config_paths['system_local'], optional=True)
            elif name.startswith('system_dir::'):
                # already handled in _load_all via merge; re-read this file
                # derive path back from name if possible
                fname = name.split('::', 1)[1]
                p = self.config_paths['system_dir'] / fname
                self._load_configuration_file(name, p, optional=True)
            elif name == 'risk':
                self._load_configuration_file('risk', self.config_paths['risk'], optional=True)
            elif name == 'explainability':
                self._load_configuration_file('explainability', self.config_paths['explainability'], optional=True)

        # Rebuild system aggregate and modules
        self._load_all_configurations()  # handles merges + module rebuilds internally

        # Notify top-level watchers on changed roots
        for cb in self.config_watchers:
            try:
                for root in ('system', 'risk', 'explainability'):
                    old_cfg = old_all.get(root, {})
                    new_cfg = self.configs.get(root, {})
                    if old_cfg != new_cfg:
                        cb(root, self._redacted(old_cfg), self._redacted(new_cfg))
            except Exception as e:
                self.logger.error(f"Error in config watcher: {e}")

        self.logger.info("Successfully reloaded configuration bundle")

    # ─────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────
    def _redacted(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact obvious secrets before logging/watching."""
        def _redact_value(obj: Any) -> Any:
            # Recursively redact values, preserving containers
            if isinstance(obj, dict):
                return {k: _redact_value(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_redact_value(x) for x in obj]
            return obj

        src: Dict[str, Any] = copy.deepcopy(data)
        redacted: Dict[str, Any] = {}
        for k, v in src.items():
            vk = k.lower()
            if any(key in vk for key in self._SECRET_KEYS):
                redacted[k] = "***redacted***"
            else:
                redacted[k] = _redact_value(v)
        return redacted

    def __del__(self):
        """Cleanup on destruction"""
        try:
            self.stop_monitoring()
        except Exception:
            pass
