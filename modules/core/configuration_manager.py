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
            category='features',
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
            config_section='modules.RiskManager.config',
            default_config={
                'risk_assessment_frequency': 60,
                'alert_notification': True,
                'automated_responses': True,
                'risk_reporting': True,
                'timeout_ms': 50
            },
            required_keys=['risk_assessment_frequency'],
            validation_rules={
                'risk_assessment_frequency': lambda x: x > 0
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

        # Additional Risk Management Modules
        self.module_specs['DynamicRiskController'] = ModuleConfigSpec(
            name='DynamicRiskController',
            category='risk',
            config_section='modules.DynamicRiskController.config',
            default_config={
                'base_risk_scale': 1.0,
                'min_risk_scale': 0.1,
                'max_risk_scale': 1.5,
                'vol_history_len': 30,
                'dd_threshold': 0.15,
                'vol_ratio_threshold': 2.0,
                'recovery_speed': 0.15,
                'risk_decay': 0.95,
                'adaptive_scaling': True,
                'regime_sensitivity': 1.0,
                'max_processing_time_ms': 100,
                'timeout_ms': 3000
            },
            required_keys=['base_risk_scale', 'dd_threshold'],
            validation_rules={
                'base_risk_scale': lambda x: x > 0,
                'dd_threshold': lambda x: 0 < x < 1,
                'vol_ratio_threshold': lambda x: x > 0
            }
        )

        self.module_specs['EnhancedAnomalyDetector'] = ModuleConfigSpec(
            name='EnhancedAnomalyDetector',
            category='risk',
            config_section='modules.EnhancedAnomalyDetector.config',
            default_config={
                'anomaly_threshold': 0.7,
                'lookback_window': 50,
                'sensitivity': 0.8,
                'detection_methods': ['statistical', 'ml_based', 'pattern'],
                'alert_cooldown': 300,
                'timeout_ms': 200
            },
            required_keys=['anomaly_threshold'],
            validation_rules={
                'anomaly_threshold': lambda x: 0 < x < 1,
                'lookback_window': lambda x: x > 0
            }
        )

        self.module_specs['PortfolioRiskSystem'] = ModuleConfigSpec(
            name='PortfolioRiskSystem',
            category='risk',
            config_section='modules.PortfolioRiskSystem.config',
            default_config={
                'max_portfolio_var': 0.025,
                'position_limit_pct': 0.15,
                'correlation_threshold': 0.8,
                'var_confidence': 0.95,
                'rebalance_threshold': 0.05,
                'timeout_ms': 150
            },
            required_keys=['max_portfolio_var'],
            validation_rules={
                'max_portfolio_var': lambda x: 0 < x < 1,
                'position_limit_pct': lambda x: 0 < x < 1
            }
        )

        self.module_specs['ExecutionQualityMonitor'] = ModuleConfigSpec(
            name='ExecutionQualityMonitor',
            category='risk',
            config_section='modules.ExecutionQualityMonitor.config',
            default_config={
                'slippage_threshold': 0.001,
                'latency_threshold_ms': 50,
                'quality_score_threshold': 0.8,
                'monitoring_window': 100,
                'alert_threshold': 0.7,
                'timeout_ms': 100
            },
            required_keys=['slippage_threshold'],
            validation_rules={
                'slippage_threshold': lambda x: x > 0,
                'quality_score_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['ActiveTradeMonitor'] = ModuleConfigSpec(
            name='ActiveTradeMonitor',
            category='risk',
            config_section='modules.ActiveTradeMonitor.config',
            default_config={
                'max_trade_duration': 3600,
                'position_check_interval': 60,
                'risk_alert_threshold': 0.8,
                'duration_alert_hours': 24,
                'timeout_ms': 100
            },
            required_keys=['max_trade_duration'],
            validation_rules={
                'max_trade_duration': lambda x: x > 0,
                'position_check_interval': lambda x: x > 0
            }
        )

        self.module_specs['ComplianceModule'] = ModuleConfigSpec(
            name='ComplianceModule',
            category='risk',
            config_section='modules.ComplianceModule.config',
            default_config={
                'max_leverage': 10.0,
                'position_limits': {'individual': 0.1, 'sector': 0.3},
                'compliance_checks': ['leverage', 'position_size', 'regulatory'],
                'alert_threshold': 0.9,
                'timeout_ms': 150
            },
            required_keys=['max_leverage'],
            validation_rules={
                'max_leverage': lambda x: x > 0,
                'alert_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['DrawdownRescue'] = ModuleConfigSpec(
            name='DrawdownRescue',
            category='risk',
            config_section='modules.DrawdownRescue.config',
            default_config={
                'rescue_threshold': 0.10,
                'max_drawdown_limit': 0.20,
                'recovery_target': 0.05,
                'rescue_mode_duration': 3600,
                'timeout_ms': 200
            },
            required_keys=['rescue_threshold', 'max_drawdown_limit'],
            validation_rules={
                'rescue_threshold': lambda x: 0 < x < 1,
                'max_drawdown_limit': lambda x: 0 < x < 1
            }
        )

        self.module_specs['CorrelatedRiskController'] = ModuleConfigSpec(
            name='CorrelatedRiskController',
            category='risk',
            config_section='modules.CorrelatedRiskController.config',
            default_config={
                'correlation_threshold': 0.7,
                'lookback_period': 60,
                'diversification_target': 0.3,
                'cluster_threshold': 0.8,
                'timeout_ms': 200
            },
            required_keys=['correlation_threshold'],
            validation_rules={
                'correlation_threshold': lambda x: 0 < x < 1,
                'diversification_target': lambda x: 0 < x < 1
            }
        )

        # Voting System Modules
        self.module_specs['StrategyArbiter'] = ModuleConfigSpec(
            name='StrategyArbiter',
            category='voting',
            config_section='modules.StrategyArbiter.config',
            default_config={
                'action_dim': 4,
                'adapt_rate': 0.01,
                'min_confidence': 0.3,
                'bootstrap_steps': 50,
                'debug': True,
                'reinforce_lr': 0.001,
                'prior_blend': 0.30,
                'timeout_ms': 3000
            },
            required_keys=['action_dim'],
            validation_rules={
                'action_dim': lambda x: x > 0,
                'min_confidence': lambda x: 0 < x < 1,
                'adapt_rate': lambda x: x > 0
            }
        )

        self.module_specs['AlternativeRealitySampler'] = ModuleConfigSpec(
            name='AlternativeRealitySampler',
            category='voting',
            config_section='modules.AlternativeRealitySampler.config',
            default_config={
                'num_samples': 1000,
                'confidence_threshold': 0.6,
                'diversity_weight': 0.3,
                'sample_horizon': 20,
                'bootstrap_ratio': 0.8,
                'timeout_ms': 6000
            },
            required_keys=['num_samples'],
            validation_rules={
                'num_samples': lambda x: x > 0,
                'confidence_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['CollusionAuditor'] = ModuleConfigSpec(
            name='CollusionAuditor',
            category='voting',
            config_section='modules.CollusionAuditor.config',
            default_config={
                'detection_threshold': 0.8,
                'correlation_window': 30,
                'independence_threshold': 0.5,
                'audit_frequency': 10,
                'suspicious_threshold': 0.7,
                'timeout_ms': 6000
            },
            required_keys=['detection_threshold'],
            validation_rules={
                'detection_threshold': lambda x: 0 < x < 1,
                'independence_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['ConsensusDetector'] = ModuleConfigSpec(
            name='ConsensusDetector',
            category='voting',
            config_section='modules.ConsensusDetector.config',
            default_config={
                'consensus_threshold': 0.7,
                'quality_threshold': 0.6,
                'member_weight_decay': 0.95,
                'confidence_weight': 0.4,
                'timeout_ms': 1000
            },
            required_keys=['consensus_threshold'],
            validation_rules={
                'consensus_threshold': lambda x: 0 < x < 1,
                'quality_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['TimeHorizonAligner'] = ModuleConfigSpec(
            name='TimeHorizonAligner',
            category='voting',
            config_section='modules.TimeHorizonAligner.config',
            default_config={
                'alignment_threshold': 0.8,
                'horizon_weights': [1.0, 0.8, 0.6, 0.4],
                'adaptation_rate': 0.05,
                'regime_sensitivity': 1.2,
                'timeout_ms': 6000
            },
            required_keys=['alignment_threshold'],
            validation_rules={
                'alignment_threshold': lambda x: 0 < x < 1,
                'adaptation_rate': lambda x: x > 0
            }
        )

        self.module_specs['VotingKernel'] = ModuleConfigSpec(
            name='VotingKernel',
            category='voting',
            config_section='modules.VotingKernel.config',
            default_config={
                'voting_method': 'weighted_average',
                'min_votes': 3,
                'confidence_weighting': True,
                'consensus_threshold': 0.6,
                'timeout_ms': 2000
            },
            required_keys=['voting_method'],
            validation_rules={
                'min_votes': lambda x: x > 0,
                'consensus_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['EnhancedVotingCommitteeCoordinator'] = ModuleConfigSpec(
            name='EnhancedVotingCommitteeCoordinator',
            category='voting',
            config_section='modules.EnhancedVotingCommitteeCoordinator.config',
            default_config={
                'committee_size': 5,
                'rotation_frequency': 50,
                'performance_weight': 0.6,
                'diversity_weight': 0.4,
                'timeout_ms': 3000
            },
            required_keys=['committee_size'],
            validation_rules={
                'committee_size': lambda x: x > 0,
                'performance_weight': lambda x: 0 < x < 1
            }
        )

        # Strategy Modules
        self.module_specs['BiasAuditor'] = ModuleConfigSpec(
            name='BiasAuditor',
            category='strategy',
            config_section='modules.BiasAuditor.config',
            default_config={
                'bias_detection_threshold': 0.7,
                'correction_strength': 0.5,
                'audit_window': 100,
                'bias_types': ['confirmation', 'anchoring', 'overconfidence'],
                'timeout_ms': 300
            },
            required_keys=['bias_detection_threshold'],
            validation_rules={
                'bias_detection_threshold': lambda x: 0 < x < 1,
                'correction_strength': lambda x: 0 < x < 1
            }
        )

        self.module_specs['CurriculumPlannerPlus'] = ModuleConfigSpec(
            name='CurriculumPlannerPlus',
            category='strategy',
            config_section='modules.CurriculumPlannerPlus.config',
            default_config={
                'learning_stages': ['basic', 'intermediate', 'advanced'],
                'mastery_threshold': 0.8,
                'progression_rate': 0.05,
                'stage_duration': 1000,
                'timeout_ms': 400
            },
            required_keys=['mastery_threshold'],
            validation_rules={
                'mastery_threshold': lambda x: 0 < x < 1,
                'progression_rate': lambda x: x > 0
            }
        )

        self.module_specs['ExplanationGenerator'] = ModuleConfigSpec(
            name='ExplanationGenerator',
            category='strategy',
            config_section='modules.ExplanationGenerator.config',
            default_config={
                'explanation_depth': 'detailed',
                'narrative_style': 'technical',
                'update_frequency': 60,
                'context_window': 50,
                'timeout_ms': 500
            },
            required_keys=['explanation_depth'],
            validation_rules={
                'explanation_depth': lambda x: x in ['basic', 'standard', 'detailed']
            }
        )

        self.module_specs['OpponentModeEnhancer'] = ModuleConfigSpec(
            name='OpponentModeEnhancer',
            category='strategy',
            config_section='modules.OpponentModeEnhancer.config',
            default_config={
                'mode_detection_window': 30,
                'adaptation_speed': 0.1,
                'market_modes': ['trending', 'ranging', 'volatile', 'calm'],
                'confidence_threshold': 0.7,
                'timeout_ms': 300
            },
            required_keys=['mode_detection_window'],
            validation_rules={
                'mode_detection_window': lambda x: x > 0,
                'adaptation_speed': lambda x: 0 < x < 1
            }
        )

        self.module_specs['PlaybookClusterer'] = ModuleConfigSpec(
            name='PlaybookClusterer',
            category='strategy',
            config_section='modules.PlaybookClusterer.config',
            default_config={
                'num_clusters': 10,
                'clustering_method': 'kmeans',
                'feature_dimensions': 20,
                'update_frequency': 100,
                'min_cluster_size': 5,
                'timeout_ms': 400
            },
            required_keys=['num_clusters'],
            validation_rules={
                'num_clusters': lambda x: x > 0,
                'min_cluster_size': lambda x: x > 0
            }
        )

        self.module_specs['StrategyIntrospector'] = ModuleConfigSpec(
            name='StrategyIntrospector',
            category='strategy',
            config_section='modules.StrategyIntrospector.config',
            default_config={
                'introspection_depth': 3,
                'pattern_window': 50,
                'adaptation_threshold': 0.6,
                'analysis_frequency': 25,
                'timeout_ms': 300
            },
            required_keys=['introspection_depth'],
            validation_rules={
                'introspection_depth': lambda x: x > 0,
                'adaptation_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['ThesisEvolutionEngine'] = ModuleConfigSpec(
            name='ThesisEvolutionEngine',
            category='strategy',
            config_section='modules.ThesisEvolutionEngine.config',
            default_config={
                'population_size': 20,
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'evolution_generations': 50,
                'fitness_threshold': 0.8,
                'timeout_ms': 500
            },
            required_keys=['population_size'],
            validation_rules={
                'population_size': lambda x: x > 0,
                'mutation_rate': lambda x: 0 < x < 1
            }
        )

        # Meta System Modules
        self.module_specs['MetaAgent'] = ModuleConfigSpec(
            name='MetaAgent',
            category='meta',
            config_section='modules.MetaAgent.config',
            default_config={
                'window': 20,
                'profit_target': 150.0,
                'retrain_threshold': -50.0,
                'emergency_threshold': -100.0,
                'confidence_threshold': 0.7,
                'min_training_episodes': 100,
                'max_processing_time_ms': 300,
                'timeout_ms': 3000
            },
            required_keys=['profit_target', 'retrain_threshold'],
            validation_rules={
                'confidence_threshold': lambda x: 0 < x < 1,
                'window': lambda x: x > 0
            }
        )

        self.module_specs['MetaCognitivePlanner'] = ModuleConfigSpec(
            name='MetaCognitivePlanner',
            category='meta',
            config_section='modules.MetaCognitivePlanner.config',
            default_config={
                'planning_horizon': 50,
                'adaptation_rate': 0.05,
                'strategic_weight': 0.6,
                'tactical_weight': 0.4,
                'timeout_ms': 400
            },
            required_keys=['planning_horizon'],
            validation_rules={
                'planning_horizon': lambda x: x > 0,
                'adaptation_rate': lambda x: x > 0
            }
        )

        self.module_specs['MetaRLController'] = ModuleConfigSpec(
            name='MetaRLController',
            category='meta',
            config_section='modules.MetaRLController.config',
            default_config={
                'learning_rate': 0.001,
                'batch_size': 64,
                'replay_buffer_size': 10000,
                'target_update_frequency': 100,
                'exploration_rate': 0.1,
                'timeout_ms': 500
            },
            required_keys=['learning_rate'],
            validation_rules={
                'learning_rate': lambda x: x > 0,
                'batch_size': lambda x: x > 0
            }
        )

        # Memory System
        self.module_specs['UnifiedMemory'] = ModuleConfigSpec(
            name='UnifiedMemory',
            category='memory',
            config_section='modules.UnifiedMemory.config',
            default_config={
                'memory_capacity': 100000,
                'compression_ratio': 0.1,
                'retrieval_threshold': 0.7,
                'pattern_recognition_depth': 5,
                'forgetting_rate': 0.001,
                'timeout_ms': 200
            },
            required_keys=['memory_capacity'],
            validation_rules={
                'memory_capacity': lambda x: x > 0,
                'compression_ratio': lambda x: 0 < x < 1
            }
        )

        # External Data Modules
        self.module_specs['SessionManager'] = ModuleConfigSpec(
            name='SessionManager',
            category='external',
            config_section='modules.SessionManager.config',
            default_config={
                'session_duration': 3600,
                'health_check_interval': 30,
                'emergency_mode_threshold': 0.9,
                'performance_tracking': True,
                'timeout_ms': 100
            },
            required_keys=['session_duration'],
            validation_rules={
                'session_duration': lambda x: x > 0,
                'health_check_interval': lambda x: x > 0
            }
        )

        self.module_specs['MarketDataProvider'] = ModuleConfigSpec(
            name='MarketDataProvider',
            category='external',
            config_section='modules.MarketDataProvider.config',
            default_config={
                'update_frequency': 1.0,
                'data_sources': ['primary', 'backup'],
                'cache_duration': 60,
                'quality_threshold': 0.95,
                'timeout_ms': 200
            },
            required_keys=['update_frequency'],
            validation_rules={
                'update_frequency': lambda x: x > 0,
                'quality_threshold': lambda x: 0 < x < 1
            }
        )

        # Execution System
        self.module_specs['Executor'] = ModuleConfigSpec(
            name='Executor',
            category='executor',
            config_section='modules.Executor.config',
            default_config={
                'execution_timeout': 5.0,
                'order_validation': True,
                'risk_checks': True,
                'slippage_tolerance': 0.001,
                'timeout_ms': 1000
            },
            required_keys=['execution_timeout'],
            validation_rules={
                'execution_timeout': lambda x: x > 0,
                'slippage_tolerance': lambda x: x > 0
            }
        )

        # Position Management
        self.module_specs['PositionManager'] = ModuleConfigSpec(
            name='PositionManager',
            category='position',
            config_section='modules.PositionManager.config',
            default_config={
                'max_positions': 10,
                'position_size_limit': 0.1,
                'rebalance_threshold': 0.05,
                'liquidation_threshold': 0.95,
                'timeout_ms': 300
            },
            required_keys=['max_positions'],
            validation_rules={
                'max_positions': lambda x: x > 0,
                'position_size_limit': lambda x: 0 < x < 1
            }
        )

        # Trading Modes
        self.module_specs['TradingModeManager'] = ModuleConfigSpec(
            name='TradingModeManager',
            category='trading_modes',
            config_section='modules.TradingModeManager.config',
            default_config={
                'available_modes': ['conservative', 'balanced', 'aggressive'],
                'mode_switch_threshold': 0.8,
                'evaluation_window': 100,
                'default_mode': 'balanced',
                'timeout_ms': 200
            },
            required_keys=['available_modes'],
            validation_rules={
                'mode_switch_threshold': lambda x: 0 < x < 1
            }
        )

        # Reward System
        self.module_specs['RiskAdjustedReward'] = ModuleConfigSpec(
            name='RiskAdjustedReward',
            category='reward',
            config_section='modules.RiskAdjustedReward.config',
            default_config={
                'risk_penalty_weight': 0.3,
                'return_weight': 0.7,
                'sharpe_target': 1.5,
                'max_drawdown_penalty': 2.0,
                'timeout_ms': 150
            },
            required_keys=['risk_penalty_weight'],
            validation_rules={
                'risk_penalty_weight': lambda x: 0 < x < 1,
                'return_weight': lambda x: 0 < x < 1
            }
        )

        # Auditing Modules
        self.module_specs['AuditingCoordinator'] = ModuleConfigSpec(
            name='AuditingCoordinator',
            category='auditing',
            config_section='modules.AuditingCoordinator.config',
            default_config={
                'audit_frequency': 100,
                'audit_depth': 'comprehensive',
                'report_format': 'detailed',
                'alert_threshold': 0.8,
                'timeout_ms': 300
            },
            required_keys=['audit_frequency'],
            validation_rules={
                'audit_frequency': lambda x: x > 0,
                'alert_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['TradeExplanationAuditor'] = ModuleConfigSpec(
            name='TradeExplanationAuditor',
            category='auditing',
            config_section='modules.TradeExplanationAuditor.config',
            default_config={
                'explanation_required': True,
                'min_explanation_quality': 0.7,
                'audit_all_trades': True,
                'explanation_timeout': 30,
                'timeout_ms': 200
            },
            required_keys=['min_explanation_quality'],
            validation_rules={
                'min_explanation_quality': lambda x: 0 < x < 1
            }
        )

        self.module_specs['TradeThesisTracker'] = ModuleConfigSpec(
            name='TradeThesisTracker',
            category='auditing',
            config_section='modules.TradeThesisTracker.config',
            default_config={
                'thesis_validation': True,
                'tracking_window': 50,
                'thesis_score_threshold': 0.6,
                'update_frequency': 10,
                'timeout_ms': 200
            },
            required_keys=['thesis_score_threshold'],
            validation_rules={
                'thesis_score_threshold': lambda x: 0 < x < 1
            }
        )

        # Features Modules
        self.module_specs['MultiScaleFeatureEngine'] = ModuleConfigSpec(
            name='MultiScaleFeatureEngine',
            category='features',
            config_section='modules.MultiScaleFeatureEngine.config',
            default_config={
                'scales': [1, 5, 15, 60],
                'feature_dimensions': 50,
                'neural_layers': [128, 64, 32],
                'dropout_rate': 0.2,
                'timeout_ms': 300
            },
            required_keys=['scales'],
            validation_rules={
                'scales': lambda x: isinstance(x, list) and len(x) > 0,
                'dropout_rate': lambda x: 0 < x < 1
            }
        )

        # Market Analysis Modules
        self.module_specs['UnifiedMarket'] = ModuleConfigSpec(
            name='UnifiedMarket',
            category='market',
            config_section='modules.UnifiedMarket.config',
            default_config={
                'regime_detection_window': 50,
                'theme_confidence_threshold': 0.7,
                'liquidity_threshold': 0.5,
                'fractal_analysis_depth': 3,
                'timeout_ms': 400
            },
            required_keys=['regime_detection_window'],
            validation_rules={
                'regime_detection_window': lambda x: x > 0,
                'theme_confidence_threshold': lambda x: 0 < x < 1
            }
        )

        self.module_specs['FractalRegimeConfirmation'] = ModuleConfigSpec(
            name='FractalRegimeConfirmation',
            category='market',
            config_section='modules.FractalRegimeConfirmation.config',
            default_config={
                'fractal_levels': [5, 13, 34],
                'confirmation_threshold': 0.75,
                'lookback_periods': 100,
                'timeout_ms': 200
            },
            required_keys=['fractal_levels'],
            validation_rules={
                'fractal_levels': lambda x: isinstance(x, list) and len(x) > 0
            }
        )

        self.module_specs['RegimePerformanceMatrix'] = ModuleConfigSpec(
            name='RegimePerformanceMatrix',
            category='market',
            config_section='modules.RegimePerformanceMatrix.config',
            default_config={
                'regime_types': ['trending', 'ranging', 'volatile'],
                'performance_window': 200,
                'update_frequency': 20,
                'confidence_threshold': 0.8,
                'timeout_ms': 300
            },
            required_keys=['regime_types'],
            validation_rules={
                'performance_window': lambda x: x > 0
            }
        )

        self.module_specs['TimeAwareRiskScaling'] = ModuleConfigSpec(
            name='TimeAwareRiskScaling',
            category='market',
            config_section='modules.TimeAwareRiskScaling.config',
            default_config={
                'time_zones': ['asian', 'london', 'newyork'],
                'volatility_scaling': True,
                'session_risk_factors': [0.8, 1.2, 1.0],
                'overlap_boost': 1.1,
                'timeout_ms': 150
            },
            required_keys=['time_zones'],
            validation_rules={
                'overlap_boost': lambda x: x > 0
            }
        )

        # Missing Modules from Log Warnings
        self.module_specs['NewsSentimentModule'] = ModuleConfigSpec(
            name='NewsSentimentModule',
            category='external',
            config_section='modules.NewsSentimentModule.config',
            default_config={
                'sentiment_threshold': 0.6,
                'news_sources': ['reuters', 'bloomberg', 'cnbc'],
                'update_frequency': 300,
                'language_models': ['sentiment_basic'],
                'timeout_ms': 500
            },
            required_keys=['sentiment_threshold'],
            validation_rules={
                'sentiment_threshold': lambda x: 0 < x < 1,
                'update_frequency': lambda x: x > 0
            }
        )

        self.module_specs['UnifiedMarketModule'] = ModuleConfigSpec(
            name='UnifiedMarketModule',
            category='market_1',
            config_section='modules.UnifiedMarketModule.config',
            default_config={
                'analysis_depth': 'comprehensive',
                'regime_detection': True,
                'liquidity_analysis': True,
                'theme_detection': True,
                'fractal_analysis': True,
                'timeout_ms': 600
            },
            required_keys=['analysis_depth'],
            validation_rules={
                'analysis_depth': lambda x: x in ['basic', 'standard', 'comprehensive']
            }
        )

        self.module_specs['PPOLagAgent'] = ModuleConfigSpec(
            name='PPOLagAgent',
            category='meta',
            config_section='modules.PPOLagAgent.config',
            default_config={
                'learning_rate': 0.0003,
                'clip_eps': 0.2,
                'value_coeff': 0.5,
                'entropy_coeff': 0.01,
                'lag_compensation': True,
                'lag_detection_threshold': 100,
                'timeout_ms': 400
            },
            required_keys=['learning_rate'],
            validation_rules={
                'learning_rate': lambda x: x > 0,
                'lag_detection_threshold': lambda x: x > 0
            }
        )

        self.module_specs['EnhancedWorldModel'] = ModuleConfigSpec(
            name='EnhancedWorldModel',
            category='models',
            config_section='modules.EnhancedWorldModel.config',
            default_config={
                'model_complexity': 'medium',
                'prediction_horizon': 50,
                'state_dimensions': 128,
                'action_dimensions': 32,
                'learning_rate': 0.001,
                'update_frequency': 10,
                'timeout_ms': 800
            },
            required_keys=['model_complexity', 'prediction_horizon'],
            validation_rules={
                'model_complexity': lambda x: x in ['low', 'medium', 'high'],
                'prediction_horizon': lambda x: x > 0,
                'learning_rate': lambda x: x > 0
            }
        )

        self.module_specs['VisualizationInterface'] = ModuleConfigSpec(
            name='VisualizationInterface',
            category='visualization',
            config_section='modules.VisualizationInterface.config',
            default_config={
                'chart_types': ['candlestick', 'line', 'volume'],
                'update_frequency': 1.0,
                'max_data_points': 1000,
                'real_time_updates': True,
                'export_formats': ['png', 'svg'],
                'timeout_ms': 200
            },
            required_keys=['chart_types'],
            validation_rules={
                'update_frequency': lambda x: x > 0,
                'max_data_points': lambda x: x > 0
            }
        )

        self.module_specs['OpponentSimulator'] = ModuleConfigSpec(
            name='OpponentSimulator',
            category='simulation',
            config_section='modules.OpponentSimulator.config',
            default_config={
                'simulation_depth': 3,
                'opponent_types': ['aggressive', 'conservative', 'random'],
                'adaptation_rate': 0.05,
                'simulation_runs': 100,
                'confidence_threshold': 0.7,
                'timeout_ms': 1000
            },
            required_keys=['simulation_depth'],
            validation_rules={
                'simulation_depth': lambda x: x > 0,
                'adaptation_rate': lambda x: 0 < x < 1,
                'simulation_runs': lambda x: x > 0
            }
        )

        self.module_specs['RoleCoach'] = ModuleConfigSpec(
            name='RoleCoach',
            category='simulation',
            config_section='modules.RoleCoach.config',
            default_config={
                'coaching_intensity': 'medium',
                'feedback_frequency': 25,
                'performance_tracking': True,
                'improvement_suggestions': True,
                'learning_acceleration': 1.2,
                'timeout_ms': 300
            },
            required_keys=['coaching_intensity'],
            validation_rules={
                'coaching_intensity': lambda x: x in ['low', 'medium', 'high'],
                'feedback_frequency': lambda x: x > 0,
                'learning_acceleration': lambda x: x > 0
            }
        )

        self.module_specs['ShadowSimulator'] = ModuleConfigSpec(
            name='ShadowSimulator',
            category='simulation',
            config_section='modules.ShadowSimulator.config',
            default_config={
                'shadow_depth': 2,
                'parallel_simulations': 5,
                'reality_divergence_threshold': 0.3,
                'shadow_update_frequency': 10,
                'confidence_weighting': True,
                'timeout_ms': 800
            },
            required_keys=['shadow_depth'],
            validation_rules={
                'shadow_depth': lambda x: x > 0,
                'parallel_simulations': lambda x: x > 0,
                'reality_divergence_threshold': lambda x: 0 < x < 1
            }
        )

        # Missing modules from contracts.py
        self.module_specs['EnhancedSeasonalityRiskExpert'] = ModuleConfigSpec(
            name='EnhancedSeasonalityRiskExpert',
            category='voting',
            config_section='modules.EnhancedSeasonalityRiskExpert.config',
            default_config={
                'seasonal_analysis_depth': 'comprehensive',
                'historical_lookback_years': 5,
                'risk_seasonality_threshold': 0.6,
                'adaptive_weighting': True,
                'timeout_ms': 300
            },
            required_keys=['seasonal_analysis_depth'],
            validation_rules={
                'seasonal_analysis_depth': lambda x: x in ['basic', 'standard', 'comprehensive'],
                'historical_lookback_years': lambda x: x > 0
            }
        )

        self.module_specs['EnhancedThemeExpert'] = ModuleConfigSpec(
            name='EnhancedThemeExpert',
            category='voting',
            config_section='modules.EnhancedThemeExpert.config',
            default_config={
                'theme_detection_threshold': 0.7,
                'theme_categories': ['bullish', 'bearish', 'neutral', 'breakout'],
                'confidence_weighting': True,
                'adaptation_speed': 0.1,
                'timeout_ms': 250
            },
            required_keys=['theme_detection_threshold'],
            validation_rules={
                'theme_detection_threshold': lambda x: 0 < x < 1,
                'adaptation_speed': lambda x: 0 < x < 1
            }
        )

        self.module_specs['TradeMapVisualizer'] = ModuleConfigSpec(
            name='TradeMapVisualizer',
            category='visualization',
            config_section='modules.TradeMapVisualizer.config',
            default_config={
                'chart_types': ['performance', 'trade_timeline', 'risk_heatmap'],
                'update_frequency': 5.0,
                'max_trade_history': 500,
                'visualization_quality': 'high',
                'export_formats': ['png', 'svg', 'pdf'],
                'timeout_ms': 400
            },
            required_keys=['chart_types'],
            validation_rules={
                'update_frequency': lambda x: x > 0,
                'max_trade_history': lambda x: x > 0
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
            if isinstance(risk_cfg, dict):
                # Apply global risk limits and controls
                global_risk = {
                    'risk_controls': risk_cfg.get('controls', {}),
                    'risk_limits': risk_cfg.get('limits', {}),
                    'escalation_thresholds': risk_cfg.get('escalation', {})
                }
                config = _deep_merge(config, global_risk)

                # Apply module-specific risk overrides
                risk_modules = risk_cfg.get('modules', {})
                if spec.name in risk_modules:
                    config = _deep_merge(config, risk_modules[spec.name])

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
