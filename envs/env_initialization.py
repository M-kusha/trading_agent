# envs/env_initialization.py
"""
Enhanced environment initialization with InfoBus integration
Maintains backward compatibility while adding InfoBus infrastructure
"""

import io
import sys
import os
import random
import logging
import torch
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Dict, Any, List, Optional

# InfoBus infrastructure
from modules.utils.info_bus import InfoBus, InfoBusBuilder, create_info_bus
from modules.utils.audit_utils import RotatingLogger, AuditTracker
from modules.core.core import ModuleConfig

# Enhanced shared utilities with InfoBus
from .shared_utils import TradingPipeline, UnifiedRiskManager

# Auditing modules
from modules.auditing.trade_explanation_auditor import TradeExplanationAuditor
from modules.auditing.trade_thesis_tracker import TradeThesisTracker

# Feature extraction modules
from modules.features.advanced_feature_engine import AdvancedFeatureEngine
from modules.features.multiscale_feature_engine import MultiScaleFeatureEngine

# Position management module
from modules.position.position import PositionManager

# Reward shaping module
from modules.reward.risk_adjusted_reward import RiskAdjustedReward

# Market analysis modules
from modules.market.market_theme_detector import MarketThemeDetector
from modules.market.fractal_regime_confirmation import FractalRegimeConfirmation
from modules.market.liquidity_heatmap_layer import LiquidityHeatmapLayer
from modules.market.time_aware_risk_scaling import TimeAwareRiskScaling
from modules.market.regime_performance_matrix import RegimePerformanceMatrix

# Memory modules
from modules.memory.neural_memory_architect import NeuralMemoryArchitect
from modules.memory.mistake_memory import MistakeMemory
from modules.memory.memory_compressor import MemoryCompressor
from modules.memory.historical_replay_analyzer import HistoricalReplayAnalyzer
from modules.memory.playbook_memory import PlaybookMemory
from modules.memory.memory_budget_optimizer import MemoryBudgetOptimizer

# Strategy modules
from modules.strategy.playbook_clusterer import PlaybookClusterer
from modules.strategy.strategy_introspector import StrategyIntrospector
from modules.strategy.curriculum_planner_plus import CurriculumPlannerPlus
from modules.strategy.strategy_genome_pool import StrategyGenomePool
from modules.strategy.bias_auditor import BiasAuditor
from modules.strategy.opponent_mode_enhancer import OpponentModeEnhancer
from modules.strategy.thesis_evolution_engine import ThesisEvolutionEngine
from modules.strategy.explanation_generator import ExplanationGenerator

# Meta-learning modules
from modules.meta.meta_agent import MetaAgent
from modules.meta.metacognitive_planner import MetaCognitivePlanner
from modules.meta.metar_rl_controller import MetaRLController

# Simulation modules
from modules.simulation.opponent_simulator import OpponentSimulator
from modules.simulation.role_coach import RoleCoach
from modules.simulation.shadow_simulator import ShadowSimulator

# Visualization modules
from modules.visualization.visualization_interface import VisualizationInterface
from modules.visualization.trade_map_visualizer import TradeMapVisualizer

# World model module
from modules.models.world_model import RNNWorldModel

# Voting modules
from modules.voting.time_horizon_aligner import TimeHorizonAligner
from modules.voting.alternative_reality_sampler import AlternativeRealitySampler
from modules.voting.collusion_auditor import CollusionAuditor
from modules.voting.consensus_detector import ConsensusDetector
from modules.voting.strategy_arbiter import StrategyArbiter
from modules.voting.voting_wrappers import (
    ThemeExpert, SeasonalityRiskExpert,
    MetaRLExpert, TradeMonitorVetoExpert, RegimeBiasExpert
)

# Trading modes
from modules.trading_modes.trading_mode import TradingModeManager

# Risk management modules
from modules.risk.active_trade_monitor import ActiveTradeMonitor
from modules.risk.correlated_risk_controller import CorrelatedRiskController
from modules.risk.drawdown_rescue import DrawdownRescue
from modules.risk.execution_quality_monitor import ExecutionQualityMonitor
from modules.risk.anomaly_detector import AnomalyDetector
from modules.risk.portofilio_risk_system import PortfolioRiskSystem
from modules.risk.compliance import ComplianceModule
from modules.risk.dynamic_risk_controller import DynamicRiskController


class DummyExplanationGenerator:
    """Dummy explanation generator for fallback compatibility"""
    
    def __init__(self):
        pass
        
    def reset(self):
        pass
        
    def step(self, **kwargs):
        pass
        
    def get_observation_components(self):
        return np.zeros(0, dtype=np.float32)


class DummyModule:
    """Dummy module for maintaining pipeline consistency"""
    def reset(self): 
        pass
    def step(self, **kwargs): 
        pass
    def get_observation_components(self): 
        return np.zeros(6, dtype=np.float32)


def _setup_logging(self):
    """Enhanced logging setup with InfoBus integration and UTF-8 support"""
    os.makedirs(self.config.log_dir, exist_ok=True)
    os.makedirs(self.config.operator_log_dir, exist_ok=True)
    
    # Create enhanced logger with rotation
    self.logger = RotatingLogger(
        name=f"EnhancedTradingEnv_{id(self)}",
        log_path=os.path.join(self.config.log_dir, "trading_env.log"),
        max_lines=self.config.log_rotation_lines,
        operator_mode=self.config.debug
    )
    
    # Audit system
    self.audit_tracker = AuditTracker("TradingEnvironment")
    
    # InfoBus logging
    if self.config.info_bus_enabled:
        self.info_bus_logger = RotatingLogger(
            name=f"InfoBus_{id(self)}",
            log_path=os.path.join(self.config.info_bus_log_dir, "info_bus.log"),
            max_lines=self.config.log_rotation_lines,
            json_mode=True
        )
    
    self.logger.info("Enhanced logging system initialized with InfoBus support")


def _set_seeds(self, seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed(self, seed=None):
    """Set the seed for this env's random number generator(s)."""
    self._set_seeds(seed if seed is not None else self.config.init_seed)
    return [seed]


def _validate_data(self):
    """Enhanced data validation with InfoBus compatibility"""
    required_columns = {"open", "high", "low", "close"}
    
    for inst in self.instruments:
        if inst not in self.data:
            raise ValueError(f"Missing data for instrument: {inst}")
            
        for tf, df in self.data[inst].items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Data for {inst}/{tf} must be DataFrame")
                
            # Check base required columns
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing columns for {inst}/{tf}: {missing}")
            
            # Handle volume column specifically
            if 'volume' not in df.columns:
                if 'real_volume' in df.columns:
                    self.logger.info(f"Creating 'volume' from 'real_volume' for {inst}/{tf}")
                    self.data[inst][tf]['volume'] = self.data[inst][tf]['real_volume']
                else:
                    self.logger.warning(f"No volume data for {inst}/{tf}, creating dummy volume")
                    self.data[inst][tf]['volume'] = 1.0


def _initialize_modules(self):
    """Enhanced module initialization with InfoBus integration and module config"""
    
    # Create module configuration
    module_config = ModuleConfig(
        debug=self.config.debug,
        max_history=100,
        audit_enabled=True,
        log_rotation_lines=self.config.log_rotation_lines,
        health_check_interval=100,
        info_bus_enabled=self.config.info_bus_enabled
    )
    
    # Core feature extraction
    base_afe = AdvancedFeatureEngine(config=module_config)
    self.feature_engine = MultiScaleFeatureEngine(base_afe, 32, config=module_config)
    
    # Position management
    self.position_manager = PositionManager(
        initial_balance=self.config.initial_balance,
        instruments=self.instruments,
        config=module_config
    )
    self.position_manager.set_env(self)
    
    # Market analysis modules
    self.theme_detector = MarketThemeDetector(
        self.instruments, 4, 100, config=module_config
    )
    self.fractal_confirm = FractalRegimeConfirmation(100, config=module_config)
    
    action_dim = 2 * len(self.instruments)
    self.liquidity_layer = LiquidityHeatmapLayer(
        action_dim=action_dim,
        config=module_config
    )
    self.time_risk_scaler = TimeAwareRiskScaling(config=module_config)
    self.regime_matrix = RegimePerformanceMatrix(config=module_config)
    
    # Enhanced risk management with InfoBus
    risk_config = {
        'dd_limit': self.config.max_drawdown,
        'correlation_limit': self.config.max_correlation,
        'var_limit': 0.1,
        'max_positions': 10,
        'alert_cooldown': self.config.risk_alert_cooldown,
    }
    self.risk_manager = UnifiedRiskManager(risk_config, logger=self.logger)
    
    # Individual risk modules
    self.risk_controller = DynamicRiskController(config=module_config)
    self.risk_system = PortfolioRiskSystem(50, 0.2, config=module_config)
    self.compliance = ComplianceModule(config=module_config)
    
    # Reward shaping
    self.reward_shaper = RiskAdjustedReward(
        self.config.initial_balance,
        env=self,
        config=module_config
    )
    
    # Memory systems with InfoBus integration
    self.mistake_memory = MistakeMemory(config=module_config)
    self.memory_compressor = MemoryCompressor(10, 5, config=module_config)
    self.replay_analyzer = HistoricalReplayAnalyzer(10, config=module_config)
    self.playbook_memory = PlaybookMemory(config=module_config)
    self.memory_budget = MemoryBudgetOptimizer(1000, 500, 300, config=module_config)
    self.long_term_memory = NeuralMemoryArchitect(32, 4, 500, config=module_config)
    
    # Strategy systems
    self.strategy_intros = StrategyIntrospector(config=module_config)
    self.curriculum_planner = CurriculumPlannerPlus(config=module_config)
    self.playbook_clusterer = PlaybookClusterer(5, config=module_config)
    self.strategy_pool = StrategyGenomePool(20, config=module_config)
    
    # Meta-learning systems
    self.meta_agent = MetaAgent(config=module_config)
    self.meta_planner = MetaCognitivePlanner(config=module_config)
    self.bias_auditor = BiasAuditor(config=module_config)
    self.thesis_engine = ThesisEvolutionEngine(config=module_config)
    
    # Trading mode and monitoring
    self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
    self.active_monitor = ActiveTradeMonitor(config=module_config)
    self.corr_controller = CorrelatedRiskController(config=module_config)
    self.dd_rescue = DrawdownRescue(config=module_config)
    self.exec_monitor = ExecutionQualityMonitor(config=module_config)
    self.anomaly_detector = AnomalyDetector(config=module_config)
    
    # Visualization and tracking
    self.visualizer = VisualizationInterface(config=module_config)
    self.trade_map_vis = TradeMapVisualizer(config=module_config)
    self.trade_thesis = TradeThesisTracker(config=module_config)
    
    # Advanced modules
    self.world_model = RNNWorldModel(2, config=module_config)
    self.opp_enhancer = OpponentModeEnhancer(config=module_config)
    
    # Simulation modules (only in backtest mode)
    if not self.config.live_mode:
        self.shadow_sim = ShadowSimulator(config=module_config)
        self.role_coach = RoleCoach(config=module_config)
        self.opponent_sim = OpponentSimulator(config=module_config)
    else:
        self.shadow_sim = None
        self.role_coach = None
        self.opponent_sim = None
    
    self.logger.info("Enhanced modules initialized with InfoBus integration")


def _initialize_arbiter(self):
    """Enhanced strategy arbiter initialization with InfoBus integration"""
    
    # Create expert wrappers first
    self.theme_expert = ThemeExpert(self.theme_detector, self)
    self.season_expert = SeasonalityRiskExpert(self.time_risk_scaler, self)
    self.meta_rl_expert = MetaRLExpert(self.meta_rl, self)  # Will be updated later
    self.veto_expert = TradeMonitorVetoExpert(self.active_monitor, self)
    self.regime_expert = RegimeBiasExpert(self.fractal_confirm, self)
    
    # Create arbiter members list
    arbiter_members = [
        self.liquidity_layer,      # Base module
        self.position_manager,     # Core decision maker
        self.theme_expert,         # Theme-based trading
        self.season_expert,        # Seasonality adjustments
        self.meta_rl_expert,       # Meta-RL decisions
        self.veto_expert,          # Risk veto
        self.regime_expert,        # Regime-based bias
        self.risk_controller,      # Risk management
    ]
    
    # Balanced initial weights
    init_weights = [
        0.15,  # liquidity_layer
        0.20,  # position_manager (higher weight)
        0.15,  # theme_expert
        0.10,  # season_expert
        0.15,  # meta_rl_expert
        0.10,  # veto_expert
        0.10,  # regime_expert
        0.05,  # risk_controller
    ]
    
    # Enhanced strategy arbiter with InfoBus
    module_config = ModuleConfig(
        debug=self.config.debug,
        log_rotation_lines=self.config.log_rotation_lines,
        info_bus_enabled=self.config.info_bus_enabled
    )
    
    self.arbiter = StrategyArbiter(
        members=arbiter_members,
        init_weights=init_weights,
        action_dim=self.action_dim,
        consensus=self.consensus,
        horizon_aligner=self.haligner,
        config=module_config,
    )
    
    # Update committee reference
    self.committee = arbiter_members
    
    self.logger.info(f"Enhanced arbiter initialized with {len(arbiter_members)} InfoBus-integrated members")


def _initialize_dependent_modules(self):
    """Initialize modules that depend on arbiter with InfoBus integration"""
    
    module_config = ModuleConfig(
        debug=self.config.debug,
        log_rotation_lines=self.config.log_rotation_lines,
        info_bus_enabled=self.config.info_bus_enabled
    )
    
    # Initialize ExplanationGenerator with enhanced InfoBus support
    try:
        self.explainer = ExplanationGenerator(
            fractal_regime=self.fractal_confirm,
            strategy_arbiter=self.arbiter,
            config=module_config
        )
        self.logger.info("Enhanced ExplanationGenerator successfully initialized with InfoBus")
    except Exception as e:
        self.logger.error(f"Failed to initialize ExplanationGenerator: {e}")
        self.explainer = DummyExplanationGenerator()
    
    # Create enhanced pipeline with InfoBus
    self._create_pipeline()


def _create_pipeline(self):
    """Create enhanced trading pipeline with InfoBus integration"""
    
    core_modules = [
        self.feature_engine, self.compliance, self.risk_system,
        self.theme_detector, self.time_risk_scaler, self.liquidity_layer,
        self.strategy_intros, self.curriculum_planner, self.memory_budget,
        self.bias_auditor, self.opp_enhancer, self.thesis_engine,
        self.regime_matrix, self.trade_thesis,
        self.mode_manager, self.active_monitor, self.corr_controller,
        self.dd_rescue, self.exec_monitor, self.anomaly_detector,
        self.position_manager, self.fractal_confirm,
        self.trade_auditor,
        self.playbook_memory, self.meta_agent,
        self.mistake_memory, self.memory_compressor, self.replay_analyzer,
        self.playbook_clusterer, self.long_term_memory,
    ]

    # Add simulation modules based on mode
    if self.config.live_mode:
        # Add dummy modules in live mode to match training pipeline
        core_modules.append(DummyModule())
    else:
        # Add actual modules in backtest
        if self.shadow_sim: 
            core_modules.append(self.shadow_sim)
        if self.role_coach: 
            core_modules.append(self.role_coach)
        if self.opponent_sim: 
            core_modules.append(self.opponent_sim)

    # Add explainer if it's not a dummy
    if not isinstance(self.explainer, DummyExplanationGenerator):
        core_modules.append(self.explainer)

    # Filter out None modules
    active_modules = [m for m in core_modules if m is not None]
    
    # Create enhanced pipeline with InfoBus support
    pipeline_config = {
        'info_bus_enabled': self.config.info_bus_enabled,
        'debug': self.config.debug
    }
    self.pipeline = TradingPipeline(active_modules, config=pipeline_config)
    
    self.logger.info(f"Enhanced pipeline created with {len(active_modules)} InfoBus-integrated modules")


def _get_stable_observation_space(self) -> spaces.Box:
    """Get stable observation space that works with InfoBus pipeline"""
    
    # Create dummy InfoBus for initial observation
    dummy_info_bus = create_info_bus(self, step=0)
    
    # Get initial observation size
    dummy_obs = self.pipeline.step(dummy_info_bus)
    
    # Add buffer for potential growth (10% extra)
    buffer_size = int(dummy_obs.shape[0] * 0.1)
    stable_size = dummy_obs.shape[0] + buffer_size
    
    # Ensure the pipeline knows the expected size
    self.pipeline.expected_size = stable_size
    
    return spaces.Box(
        low=-np.inf, 
        high=np.inf, 
        shape=(stable_size,), 
        dtype=np.float32
    )


def _create_dummy_input(self) -> InfoBus:
    """Create dummy InfoBus for initialization"""
    return create_info_bus(self, step=0)