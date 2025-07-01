# envs/env_initialization.py

import io
import sys
import os
import random
import logging
import torch
import numpy as np
import pandas as pd
from gymnasium import spaces

from .shared_utils import DummyExplanationGenerator, TradingPipeline, UnifiedRiskManager
# Auditing modules
from modules.auditing.trade_explanation_auditor import TradeExplanationAuditor
from modules.auditing.trade_thesis_tracker import TradeThesisTracker
# feature extraction modules
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
from modules.visualization.visualization_interface import VisualizationInterface, TradeMapVisualizer
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


class DummyModule:
    def reset(self): pass
    def step(self, **kwargs): pass
    def get_observation_components(self): return np.zeros(6, dtype=np.float32)  # choose size to match expected


def _setup_logging(self):
    """Setup logging configuration with UTF-8 console support"""
    os.makedirs(self.config.log_dir, exist_ok=True)
    
    # Create logger
    self.logger = logging.getLogger(f"EnhancedTradingEnv_{id(self)}")
    self.logger.handlers.clear()
    
    # File handler (always UTF-8)
    fh = logging.FileHandler(
        os.path.join(self.config.log_dir, "trading_env.log"),
        encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    self.logger.addHandler(fh)
    
    # UTF-8 console handler
    # wrap stdout.buffer so we can force UTF-8
    utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
    ch = logging.StreamHandler(utf8_stdout)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    self.logger.addHandler(ch)
    
    self.logger.setLevel(getattr(logging, self.config.log_level))
    self.logger.propagate = False


def _set_seeds(self, seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed(self, seed=None):
    """Set the seed for this env's random number generator(s).
    Kept for backward compatibility with older gym versions."""
    self._set_seeds(seed if seed is not None else self.config.init_seed)
    return [seed]


def _validate_data(self):
    """Validate input data structure and fix column issues"""
    # FIXED: Modified to handle 'real_volume' vs 'volume' mismatch
    required_columns = {"open", "high", "low", "close"}  # Base required columns
    
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
            
            # FIXED: Handle volume column specifically
            if 'volume' not in df.columns:
                # If real_volume exists, use it for volume
                if 'real_volume' in df.columns:
                    self.logger.info(f"Creating 'volume' from 'real_volume' for {inst}/{tf}")
                    self.data[inst][tf]['volume'] = self.data[inst][tf]['real_volume']
                else:
                    # Create a dummy volume column with ones
                    self.logger.warning(f"No volume data for {inst}/{tf}, creating dummy volume")
                    self.data[inst][tf]['volume'] = 1.0


def _initialize_modules(self):
    """Initialize core trading modules (no dependencies on arbiter)"""
    # Core feature extraction
    base_afe = AdvancedFeatureEngine(debug=self.config.debug)
    self.feature_engine = MultiScaleFeatureEngine(base_afe, 32, self.config.debug)
    
    # Position management
    self.position_manager = PositionManager(
        initial_balance=self.config.initial_balance,
        instruments=self.instruments,
        debug=self.config.debug
    )
    self.position_manager.set_env(self)
    
    # Market analysis
    self.theme_detector = MarketThemeDetector(
        self.instruments, 4, 100, debug=self.config.debug
    )
    self.fractal_confirm = FractalRegimeConfirmation(100, debug=self.config.debug)
    action_dim = 2 * len(self.instruments)
    self.liquidity_layer = LiquidityHeatmapLayer(
        action_dim=action_dim,
        debug=self.config.debug
    )
    self.time_risk_scaler = TimeAwareRiskScaling(debug=self.config.debug)
    self.regime_matrix = RegimePerformanceMatrix(debug=self.config.debug)
    
    # Risk management
    self.risk_controller = DynamicRiskController(
        config={
            "freeze_counter": 0,
            "freeze_duration": 5,
            "vol_history_len": 100,
            "dd_threshold": 0.2,
            "vol_ratio_threshold": 1.5,
        },
        debug=self.config.debug
    )
    self.risk_system = PortfolioRiskSystem(50, 0.2, debug=self.config.debug)
    self.compliance = ComplianceModule()
    
    # FIXED: Centralized risk manager
    self.risk_manager = UnifiedRiskManager(
        {
            'dd_limit': 0.3,
            'correlation_limit': 0.8,
            'var_limit': 0.1,
            'max_positions': 10,
        },
        logger=self.logger
    )
    
    # Reward shaping
    self.reward_shaper = RiskAdjustedReward(
        self.config.initial_balance,
        env=self,
        debug=self.config.debug
    )
    
    # Memory systems
    self.mistake_memory = MistakeMemory(interval=10, n_clusters=3, debug=self.config.debug)
    self.memory_compressor = MemoryCompressor(10, 5, debug=self.config.debug)
    self.replay_analyzer = HistoricalReplayAnalyzer(10, debug=self.config.debug)
    self.playbook_memory = PlaybookMemory(debug=self.config.debug)
    self.memory_budget = MemoryBudgetOptimizer(1000, 500, 300, debug=self.config.debug)
    self.long_term_memory = NeuralMemoryArchitect(32, 4, 500, debug=self.config.debug)
    
    # Strategy systems
    self.strategy_intros = StrategyIntrospector(debug=self.config.debug)
    self.curriculum_planner = CurriculumPlannerPlus(debug=self.config.debug)
    self.playbook_clusterer = PlaybookClusterer(5, debug=self.config.debug)
    self.strategy_pool = StrategyGenomePool(20, debug=self.config.debug)
    
    # Meta-learning systems
    self.meta_agent = MetaAgent(debug=self.config.debug)
    self.meta_planner = MetaCognitivePlanner(debug=self.config.debug)
    self.bias_auditor = BiasAuditor(debug=self.config.debug)
    self.thesis_engine = ThesisEvolutionEngine(debug=self.config.debug)
    # NOTE: ExplanationGenerator will be created AFTER arbiter is initialized
    
    # Trading mode and monitoring
    self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
    self.active_monitor = ActiveTradeMonitor(max_duration=self.config.max_steps)
    self.corr_controller = CorrelatedRiskController(max_corr=0.8)
    self.dd_rescue = DrawdownRescue(dd_limit=0.3)
    self.exec_monitor = ExecutionQualityMonitor()
    self.anomaly_detector = AnomalyDetector()
    
    # Visualization and tracking
    self.visualizer = VisualizationInterface(debug=self.config.debug)
    self.trade_map_vis = TradeMapVisualizer(debug=self.config.debug)
    self.trade_thesis = TradeThesisTracker(debug=self.config.debug)
    
    # Advanced modules
    self.world_model = RNNWorldModel(2, debug=self.config.debug)
    self.opp_enhancer = OpponentModeEnhancer(debug=self.config.debug)
    
    # Simulation modules (only in backtest mode)
    if not self.config.live_mode:
        self.shadow_sim = ShadowSimulator(debug=self.config.debug)
        self.role_coach = RoleCoach(debug=self.config.debug)
        self.opponent_sim = OpponentSimulator(debug=self.config.debug)
    else:
        self.shadow_sim = None
        self.role_coach = None
        self.opponent_sim = None


def _initialize_arbiter(self):
    """FIXED: Initialize the strategy arbiter with ALL expert wrappers"""
    # Create expert wrappers first
    self.theme_expert = ThemeExpert(self.theme_detector, self)
    self.season_expert = SeasonalityRiskExpert(self.time_risk_scaler, self)
    self.meta_rl_expert = MetaRLExpert(self.meta_rl, self)
    self.veto_expert = TradeMonitorVetoExpert(self.active_monitor, self)
    self.regime_expert = RegimeBiasExpert(self.fractal_confirm, self)
    
    # FIXED: Include ALL voting members
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
    
    self.arbiter = StrategyArbiter(
        members=arbiter_members,
        init_weights=init_weights,
        action_dim=self.action_dim,
        consensus=self.consensus,
        horizon_aligner=self.haligner,
        debug=self.config.debug,
    )
    
    # Also update committee reference
    self.committee = arbiter_members
    
    self.logger.info(f"Initialized arbiter with {len(arbiter_members)} voting members")


def _initialize_dependent_modules(self):
    """Initialize modules that depend on arbiter - FIXED"""
    # Correctly instantiate ExplanationGenerator with required arguments
    try:
        self.explainer = ExplanationGenerator(
            fractal_regime=self.fractal_confirm,
            strategy_arbiter=self.arbiter,
            debug=self.config.debug
        )
        self.logger.info("ExplanationGenerator successfully initialized.")
    except Exception as e:
        self.logger.error(f"Failed to initialize ExplanationGenerator: {e}")
        # Create a dummy explainer as fallback
        self.explainer = DummyExplanationGenerator()
    
    # Create module pipeline with all modules
    self._create_pipeline()





def _create_pipeline(self):
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

    # --- PATCH: Always append dummy modules in live mode to match training pipeline ---
    if self.config.live_mode:
        core_modules.append(DummyModule())  # Add as many as needed

    else:
        # Add actual modules in backtest
        if self.shadow_sim: core_modules.append(self.shadow_sim)
        if self.role_coach: core_modules.append(self.role_coach)
        if self.opponent_sim: core_modules.append(self.opponent_sim)

    # Only add explainer if it's not a dummy
    if not isinstance(self.explainer, DummyExplanationGenerator):
        core_modules.append(self.explainer)

    active_modules = [m for m in core_modules if m is not None]
    self.pipeline = TradingPipeline(active_modules)
    self.logger.info(f"Created pipeline with {len(active_modules)} active modules")



def _get_stable_observation_space(self) -> spaces.Box:
    """Get a stable observation space that won't change during training"""
    # Create dummy observation to get initial size
    dummy_obs = self._get_full_observation(self._create_dummy_input())
    
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


def _create_dummy_input(self) -> dict:
    """Create dummy input for initialization"""
    return {
        "env": self,
        "price_h1": np.zeros(7, dtype=np.float32),
        "price_h4": np.zeros(7, dtype=np.float32),
        "price_d1": np.zeros(7, dtype=np.float32),
        "actions": np.zeros(self.action_dim, dtype=np.float32),
        "trades": [],
        "open_positions": [],  # FIXED: Standardized naming
        "drawdown": 0.0,
        "memory": np.zeros(32, dtype=np.float32),
        "pnl": 0.0,
        "correlations": {},
        "current_step": 0,  # FIXED: Added for monitor
    }