# envs/env.py
"""
State-of-the-art PPO Trading Environment
FIXED: Proper data flow, committee wiring, and live trade recording
"""
from __future__ import annotations


# ─────────────────────────── Std-lib ──────────────────────────────────
import io
from itertools import combinations
from dataclasses import dataclass, field
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math
import os
import copy
import random
import logging
import pickle
import warnings
import time
from functools import wraps

# ───────────────────────── Third-party ────────────────────────────────
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
import MetaTrader5 as mt5

# ──────────────────────── Internal modules ────────────────────────────
from modules.auditing.explanation_auditor import TradeExplanationAuditor
from modules.core.core import Module
from modules.features.feature import AdvancedFeatureEngine, MultiScaleFeatureEngine
from modules.position.position import PositionManager
from modules.regime.regime import MarketRegimeSwitcher
from modules.reward.reward import RiskAdjustedReward
from modules.risk.risk_controller import DynamicRiskController
from modules.market.market import (
    MarketThemeDetector, FractalRegimeConfirmation,
    TimeAwareRiskScaling, LiquidityHeatmapLayer,
    RegimePerformanceMatrix,
)
from modules.memory.memory import (
    MistakeMemory, MemoryCompressor,
    HistoricalReplayAnalyzer, PlaybookMemory,
    MemoryBudgetOptimizer,
)
from modules.strategy.playbook import PlaybookClusterer
from modules.strategy.strategy import (
    StrategyIntrospector, CurriculumPlannerPlus,
    StrategyGenomePool, MetaAgent, MetaCognitivePlanner,
    BiasAuditor, OpponentModeEnhancer,
    ThesisEvolutionEngine, ExplanationGenerator, MetaRLController,
)
from modules.memory.architecture import NeuralMemoryArchitect
from modules.simulation.opponent import OpponentSimulator
from modules.simulation.simulation import ShadowSimulator, RoleCoach
from modules.visualization.visualization import VisualizationInterface, TradeMapVisualizer
from modules.auditing.tracker import TradeThesisTracker
from modules.models.world_model import RNNWorldModel
from modules.risk.compliance import ComplianceModule
from modules.risk.portfolio import PortfolioRiskSystem
from modules.strategy.voting import (
    ConsensusDetector, CollusionAuditor,
    TimeHorizonAligner, AlternativeRealitySampler,
    StrategyArbiter,
)
from modules.strategy.voting_wrappers import (
    ThemeExpert, SeasonalityRiskExpert,
    MetaRLExpert, TradeMonitorVetoExpert, RegimeBiasExpert
)
from modules.trading_modes.trading_mode import TradingModeManager
from modules.risk.risk_monitor import (
    ActiveTradeMonitor, CorrelatedRiskController, DrawdownRescue,
    ExecutionQualityMonitor, AnomalyDetector,
)

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                         Configuration Classes                        ║
# ╚═════════════════════════════════════════════════════════════════════╝

@dataclass
class TradingConfig:
    """Centralized configuration for the trading environment"""
    # Core parameters
    initial_balance: float = 3000.0
    max_steps: int = 200
    debug: bool = True
    init_seed: int = 0
    checkpoint_dir: str = "checkpoints"
    max_steps_per_episode: int = field(init=False)
    
    # Trading parameters
    no_trade_penalty: float = 0.3
    consensus_min: float = 0.30
    consensus_max: float = 0.70
    max_episodes: int = 10000
    
    # Risk parameters
    min_intensity: float = 0.25
    min_inst_confidence: float = 0.60
    rotation_gap: int = 5
    max_position_pct: float = 0.10
    max_total_exposure: float = 0.30
    
    # Module flags
    live_mode: bool = False
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        # create the alias *after* dataclass finishes init
        object.__setattr__(self, "max_steps_per_episode", self.max_steps)


@dataclass
class MarketState:
    """Encapsulates current market state"""
    balance: float
    peak_balance: float
    current_step: int
    current_drawdown: float
    # FIXED: Removed open_positions - use position_manager.open_positions as single source of truth
    last_trade_step: Dict[str, int] = field(default_factory=dict)
    

@dataclass
class EpisodeMetrics:
    """Tracks episode-level metrics"""
    pnls: List[float] = field(default_factory=list)
    durations: List[int] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    votes_log: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Unified Risk Manager                          ║
# ╚═════════════════════════════════════════════════════════════════════╝

class UnifiedRiskManager:
    """FIXED: Centralized risk management system"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.dd_limit = config.get('dd_limit', 0.3)
        self.correlation_limit = config.get('correlation_limit', 0.8)
        self.var_limit = config.get('var_limit', 0.1)
        self.max_positions = config.get('max_positions', 10)
        self.logger = logger or logging.getLogger("UnifiedRiskManager")
        
    def pre_trade_check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Centralized pre-trade risk checks"""
        # Check drawdown
        if context['drawdown'] > self.dd_limit:
            return False, f"Drawdown {context['drawdown']:.1%} exceeds limit {self.dd_limit:.1%}"
            
        # Check correlations
        max_corr = max(context.get('correlations', {}).values()) if context.get('correlations') else 0
        if max_corr > self.correlation_limit:
            return False, f"Correlation {max_corr:.2f} exceeds limit {self.correlation_limit:.2f}"
            
        # Check position count
        if len(context.get('open_positions', {})) >= self.max_positions:
            return False, f"Already have {len(context['open_positions'])} positions (max: {self.max_positions})"
            
        return True, "All risk checks passed"
        
    def post_trade_update(self, trade: Dict[str, Any]):
        """Update risk systems after trade"""
        # This would update internal risk tracking
        pass


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                     Processing Pipeline Wrapper                      ║
# ╚═════════════════════════════════════════════════════════════════════╝

def profile_method(func):
    """Performance profiling decorator"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.1 and hasattr(self, 'logger'):  # Log slow operations
            self.logger.warning(f"{func.__name__} took {elapsed:.3f}s")
            
        return result
    return wrapper


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                     Processing Pipeline Wrapper                      ║
# ╚═════════════════════════════════════════════════════════════════════╝
class TradingPipeline:
    """Manages the sequential processing of trading modules"""

    def __init__(self, modules: List[Module]):
        self.modules = modules
        self._module_map = {m.__class__.__name__: m for m in modules}

        # --- new -------------------------------------------------------
        self.expected_size: Optional[int] = None  # length the model already saw
        # ---------------------------------------------------------------

    def reset(self):
        for module in self.modules:
            try:
                module.reset()
            except Exception as e:
                logging.warning(f"Failed to reset {module.__class__.__name__}: {e}")



    # in modules below TradingPipeline class
    def step(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Run each module.step(), collect its observation components, then
        concatenate (with padding/truncation) into a fixed‐size vector.
        """
        env = data.get("env")
        obs_parts: List[np.ndarray] = []

        for module in self.modules:
            # respect per‐module enable/disable
            if env and not env.module_enabled.get(module.__class__.__name__, True):
                continue

            try:
                # build kwargs from signature
                sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
                kwargs = {k: data[k] for k in sig if k in data}
                module.step(**kwargs)
                part = module.get_observation_components()
            except Exception as e:
                logging.error(f"Error in {module.__class__.__name__}.step(): {e}")
                part = np.zeros(0, dtype=np.float32)

            # ensure 1D float32
            part = np.asarray(part, dtype=np.float32).ravel()

            # ←— New sanity check: crash into pdb if any NaN/Inf
            if not np.all(np.isfinite(part)):
                logging.error(f"🛑 {module.__class__.__name__} output contained NaN/Inf: {part}")
                # dump last inputs for this module
                for k in kwargs:
                    v = kwargs[k]
                    if isinstance(v, np.ndarray) and not np.all(np.isfinite(v)):
                        logging.error(f"    input '{k}' has bad values: {v}")
                import pdb; pdb.set_trace()
            # ————————————————————————————————

            obs_parts.append(part)

        # concatenate or default to empty
        obs = np.concatenate(obs_parts) if obs_parts else np.zeros(0, np.float32)

        # on first call, lock in expected size; afterwards pad/truncate
        if self.expected_size is None:
            self.expected_size = obs.size
        else:
            if obs.size != self.expected_size:
                if obs.size < self.expected_size:
                    pad = np.zeros(self.expected_size - obs.size, dtype=np.float32)
                    obs = np.concatenate([obs, pad])
                else:
                    obs = obs[: self.expected_size]

        return obs


    def get_module(self, name: str) -> Optional[Module]:
        return self._module_map.get(name)



# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Enhanced Trading Environment                  ║
# ╚═════════════════════════════════════════════════════════════════════╝

class EnhancedTradingEnv(gym.Env):
    """State-of-the-art trading environment with robust module integration"""

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[TradingConfig] = None,
    ):
        super().__init__()

        # ─── Core placeholders ───────────────────────────────────────
        self.committee = []                  # filled by arbiter
        self._obs_cache = {}                 # for cached observations
        self.current_step = 0
        self.module_enabled = defaultdict(lambda: True)

        # ─── Config, logging, seeds ─────────────────────────────────
        self.config = config or TradingConfig()
        self._setup_logging()
        self._set_seeds(self.config.init_seed)

        # ─── Market & episode state ─────────────────────────────────
        initial_bal = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_bal,
            peak_balance=initial_bal,
            current_step=0,
            current_drawdown=0.0,
        )
        self.episode_count = 0
        self.episode_metrics = EpisodeMetrics()

        # ─── Data & instruments ──────────────────────────────────────
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(self.data.keys())
        self._validate_data()

        # ─── Action space ───────────────────────────────────────────
        self.action_dim = 2 * len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )
        self.trade_auditor = TradeExplanationAuditor(
            history_len=200, debug=self.config.debug
        )

        # ─── Stub out meta_rl so arbiter creation won't break ────────
        self.meta_rl = None

        # ─── Consensus & horizon‐aligner for arbiter ────────────────
        self.consensus = ConsensusDetector(0)  # will resize after arbiter
        self.haligner = TimeHorizonAligner([1, 4, 24, 96])
        # (collusion isn’t consumed by arbiter, but env.step() might use it)
        self.collusion = CollusionAuditor(4, 3, debug=self.config.debug)

        # ─── Initialize all core modules ────────────────────────────
        self._initialize_modules()

        # ─── Wire up strategy arbiter & pipeline ────────────────────
        self._initialize_arbiter()            # uses self.consensus & self.haligner
        self._initialize_dependent_modules()  # builds self.pipeline

        # ─── Now that committee is known, resize consensus properly ──
        self.consensus.resize(len(self.committee))

        # ─── Create stable observation space (pipeline must exist) ──
        self.observation_space = self._get_stable_observation_space()
        obs_dim = self.observation_space.shape[0]

        # ─── Instantiate the real MetaRL controller ─────────────────
        self.meta_rl = MetaRLController(
            obs_dim, self.action_dim, debug=self.config.debug
        )

        # ─── Replace the stub MetaRLExpert inside the arbiter ───────
        from modules.strategy.voting_wrappers import MetaRLExpert
        for idx, member in enumerate(self.arbiter.members):
            if isinstance(member, MetaRLExpert):
                self.arbiter.members[idx] = MetaRLExpert(self.meta_rl, self)
                break

        # ─── Final trading‐state fields ─────────────────────────────
        self.trades: List[Dict[str, Any]] = []
        self.current_genome = None
        self._last_actions = np.zeros(self.action_dim, dtype=np.float32)
        self._last_reward = 0.0
        self.info_bus = None
        self.point_value = {
            "EUR/USD": 100000,
            "XAU/USD": 100,
            "EURUSD": 100000,
            "XAUUSD": 100,
        }

        self.logger.info(
            f"Environment initialized with {len(self.instruments)} instruments, "
            f"action_dim={self.action_dim}, obs_dim={obs_dim}"
        )

    # ═══════════════════════════════════════════════════════════════════
    #  Initialization Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(f"EnhancedTradingEnv_{id(self)}")
        self.logger.handlers.clear()
        
        # File handler
        fh = logging.FileHandler(
            os.path.join(self.config.log_dir, "trading_env.log")
        )
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        
        self.logger.addHandler(fh)
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
        self.regime_switcher = MarketRegimeSwitcher(debug=self.config.debug)
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
        self.memory_compressor = MemoryCompressor(50, 5, debug=self.config.debug)
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

        
        
        
    def _create_pipeline(self):
        """FIXED: Create the processing pipeline with all active modules"""
        core_modules = [
            self.feature_engine, self.compliance, self.risk_system,
            self.theme_detector, self.time_risk_scaler, self.liquidity_layer,
            self.strategy_intros, self.curriculum_planner, self.memory_budget,
            self.bias_auditor, self.opp_enhancer, self.thesis_engine,
            self.regime_matrix, self.trade_thesis,
            self.mode_manager, self.active_monitor, self.corr_controller,
            self.dd_rescue, self.exec_monitor, self.anomaly_detector,
            self.position_manager, self.fractal_confirm, self.regime_switcher,
            self.trade_auditor,
            # FIXED: Added missing modules
            self.playbook_memory, self.meta_agent, self.explainer,
            self.mistake_memory, self.memory_compressor, self.replay_analyzer,
            self.playbook_clusterer, self.long_term_memory,
        ]
        
        # Add simulation modules if not in live mode
        if not self.config.live_mode:
            core_modules.extend([self.shadow_sim, self.role_coach, self.opponent_sim])
            
        # Filter out None modules
        active_modules = [m for m in core_modules if m is not None]
        
        self.pipeline = TradingPipeline(active_modules)
        
        
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
        """Initialize modules that depend on arbiter"""
        # NOW we can create ExplanationGenerator with the arbiter
        self.explainer = ExplanationGenerator(
            regime_switcher=self.regime_switcher,
            strategy_arbiter=self.arbiter,
            debug=self.config.debug
        )
        
        # Create module pipeline with all modules
        self._create_pipeline()

        
    def _create_dummy_input(self) -> Dict[str, Any]:
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
        
    # ═══════════════════════════════════════════════════════════════════
    #  Core Environment Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            self._set_seeds(seed)
            
        self.logger.info(f"Resetting environment for episode {self.episode_count + 1}")
        
        # Reset episode tracking
        self.episode_count += 1
        self.episode_metrics = EpisodeMetrics()
        
        # Reset data
        self.data = copy.deepcopy(self.orig_data)
        
        # Reset market state
        initial_balance = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_balance,
            peak_balance=initial_balance,
            current_step=self._select_starting_step(),
            current_drawdown=0.0,
            last_trade_step={inst: -999 for inst in self.instruments}
        )
        self.current_step = self.market_state.current_step  # FIXED: Sync step counters
        
        # Clear caches
        if not hasattr(self, '_obs_cache') or not isinstance(self._obs_cache, dict):
            self._obs_cache = {}
        else:
            self._obs_cache.clear()
        
        # Reset all modules
        self._reset_all_modules()
        
        # Prime risk system with historical data
        self._prime_risk_system()
        
        # Select new strategy genome
        self._select_strategy_genome()
        
        # Get initial observation
        obs = self._get_full_observation(self._create_dummy_input())
        obs = self._sanitize_observation(obs)
        
        # Initialize meta-RL embedding
        if hasattr(self.meta_rl, 'last_embedding'):
            self.meta_rl.last_embedding = np.zeros_like(obs)
            
        info = self._create_reset_info()
        
        return obs, info
        
    @profile_method
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        try:
            # Validate and sanitize actions
            actions = self._validate_actions(actions)
            
            # Clear previous step data
            self.trades = []
            
            # Update position manager
            self.position_manager.step()
            
            # Apply meta-RL overlay
            actions = self._apply_meta_rl(actions)
            
            # Perform risk checks
            if not self._pass_risk_checks():
                return self._create_no_trade_step(actions)
                
            # Get committee votes and blend actions
            actions = self._get_committee_decision(actions)
            
            # Check consensus threshold
            consensus = self._calculate_consensus()
            if not self._pass_consensus_check(consensus):
                return self._create_no_trade_step(actions)
                
            # Execute trades
            trades = self._execute_trades(actions)
            
            # Update state and calculate reward
            obs, reward, terminated, truncated, info = self._finalize_step(
                trades, actions, consensus
            )
            
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            self.logger.exception(f"Error in step(): {e}")
            # Return safe default values
            obs = self._get_full_observation(self._create_dummy_input())
            return obs, -1.0, True, False, {"error": str(e)}
            
    # ═══════════════════════════════════════════════════════════════════
    #  Step Implementation Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Validate and sanitize action array"""
        # Convert to numpy array
        actions = np.asarray(actions, dtype=np.float32)
        
        # Ensure correct shape
        if actions.shape != (self.action_dim,):
            self.logger.warning(
                f"Invalid action shape {actions.shape}, expected {(self.action_dim,)}"
            )
            actions = actions.reshape(-1)[:self.action_dim]
            
        # Clip to valid range
        actions = np.clip(actions, -1.0, 1.0)
        
        # Replace any NaN/Inf
        actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return actions
        
    def _apply_meta_rl(self, actions: np.ndarray) -> np.ndarray:
        """Apply meta-RL modulation to actions"""
        if hasattr(self.meta_rl, 'modulate_action'):
            try:
                modulated = self.meta_rl.modulate_action(actions)
                return np.asarray(modulated, dtype=np.float32)
            except Exception as e:
                self.logger.warning(f"Meta-RL modulation failed: {e}")
        return actions
        
    def _pass_risk_checks(self) -> bool:
        """FIXED: Simplified risk check using centralized manager"""
        context = {
            'drawdown': self.market_state.current_drawdown,
            'correlations': self.get_instrument_correlations(),
            'open_positions': self.position_manager.open_positions,
            'returns': self._get_recent_returns(),
        }
        
        passed, reason = self.risk_manager.pre_trade_check(context)
        if not passed:
            self.logger.info(f"Risk check failed: {reason}")
            
        # Also update legacy risk modules for compatibility
        vol = self._get_current_volatility()
        self.risk_controller.adjust_risk({
            "drawdown": self.market_state.current_drawdown,
            "volatility": vol
        })
        
        return passed
        
    @profile_method
    def _get_committee_decision(self, actions: np.ndarray) -> np.ndarray:
        """
        FIXED: Enhanced committee decision with proper data flow to regime modules
        """
        votes_by_sym_tf = {}
        blended_by_sym_tf = {}
        committee_names = [m.__class__.__name__ for m in self.arbiter.members]
        
        # FIXED: Get current market data for regime detection
        current_price = None
        volatility = 0.01
        
        try:
            inst = self.instruments[0]
            df = self.data[inst]["D1"]
            step = self.market_state.current_step
            
            if step < len(df):
                current_price = float(df.iloc[step]["close"])
                
                # Calculate volatility from recent prices
                if step >= 20:
                    recent_prices = df["close"].iloc[max(0, step-20):step+1].values
                    returns = np.diff(recent_prices) / recent_prices[:-1]
                    volatility = float(np.std(returns[np.isfinite(returns)]))
                    
        except Exception as e:
            self.logger.warning(f"Failed to extract market data: {e}")
            current_price = 2000.0
            volatility = 0.01
        
        # FIXED: Update regime switcher with current price before committee decision
        try:
            self.regime_switcher.step(
                price=current_price,
                data_dict=self.data,
                current_step=self.market_state.current_step
            )
            
            # Also update fractal confirmation
            self.fractal_confirm.step(
                data_dict=self.data,
                current_step=self.market_state.current_step,
                theme_detector=self.theme_detector
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update regime modules: {e}")
        
        # Collect votes for each instrument/timeframe combination
        for inst in self.instruments:
            for tf in ["H1", "H4", "D1"]:
                # Get price history
                hist = self._get_price_history(inst, tf)
                
                # Create comprehensive observation for this timeframe
                obs_data = {
                    "env": self,
                    "price_h1": hist if tf == "H1" else np.zeros_like(hist),
                    "price_h4": hist if tf == "H4" else np.zeros_like(hist),
                    "price_d1": hist if tf == "D1" else np.zeros_like(hist),
                    "actions": actions,
                    "current_step": self.market_state.current_step,
                    "data_dict": self.data,
                    "price": current_price,
                    "volatility": volatility,
                    "balance": self.market_state.balance,
                    "drawdown": self.market_state.current_drawdown,
                }
                obs = self._get_full_observation(obs_data)
                
                # Get arbiter's blended proposal
                blend = self.arbiter.propose(obs)
                
                # Store votes
                if self.arbiter.last_alpha is not None:
                    alpha = self.arbiter.last_alpha.copy()
                else:
                    alpha = np.zeros(len(self.arbiter.members))
                    
                votes_by_sym_tf[(inst, tf)] = dict(zip(committee_names, alpha.tolist()))
                blended_by_sym_tf[(inst, tf)] = blend
                
        # Store votes for logging
        self.episode_metrics.votes_log.append(votes_by_sym_tf)
        
        # Blend across timeframes for final action
        final_action = np.zeros(self.action_dim, dtype=np.float32)
        weights = {"H1": 0.3, "H4": 0.4, "D1": 0.3}
        
        for i, inst in enumerate(self.instruments):
            intensity_sum = 0.0
            duration_sum = 0.0
            total_weight = 0.0
            
            for tf in ["H1", "H4", "D1"]:
                blend = blended_by_sym_tf[(inst, tf)]
                w = weights[tf]
                intensity_sum += blend[2*i] * w
                duration_sum += blend[2*i+1] * w
                total_weight += w
                
            if total_weight > 0:
                final_action[2*i] = intensity_sum / total_weight
                final_action[2*i+1] = duration_sum / total_weight
                
        return final_action

        
    def _calculate_consensus(self) -> float:
        """Calculate consensus level from committee votes"""
        if not hasattr(self.arbiter, 'last_alpha') or self.arbiter.last_alpha is None:
            return 0.5
            
        # Use coefficient of variation as consensus measure
        alpha = self.arbiter.last_alpha
        if alpha.sum() > 0:
            normalized = alpha / (alpha.sum() + 1e-12)
            entropy = -np.sum(normalized * np.log(normalized + 1e-12))
            max_entropy = np.log(len(alpha))
            consensus = 1.0 - (entropy / max_entropy)
        else:
            consensus = 0.0
            
        return float(consensus)
        
    def _pass_consensus_check(self, consensus: float) -> bool:
        """Check if consensus meets threshold"""
        # More lenient check
        if consensus < self.config.consensus_min:
            self.logger.debug(f"Low consensus: {consensus:.3f}")
            return np.random.random() < 0.3  # 30% chance to trade anyway
            
        if consensus > self.config.consensus_max:
            self.logger.debug(f"High consensus: {consensus:.3f}")
            
        return True
        
    def _execute_trades(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """Execute trades based on actions"""
        trades = []
        
        for i, instrument in enumerate(self.instruments):
            intensity = float(actions[2*i])
            duration_norm = float(actions[2*i+1])
            
            # Skip if intensity too low
            if abs(intensity) < self.config.min_intensity:
                continue
                
            # Check rotation gap
            last_step = self.market_state.last_trade_step.get(instrument, -999)
            if self.market_state.current_step - last_step < self.config.rotation_gap:
                continue
                
            # Check instrument confidence
            inst_conf = self.position_manager.position_confidence.get(instrument, 1.0)
            if inst_conf < self.config.min_inst_confidence:
                self.logger.debug(
                    f"Skipping {instrument}: confidence {inst_conf:.3f} < {self.config.min_inst_confidence}"
                )
                continue
                
            # Execute trade
            trade = self._execute_single_trade(instrument, intensity, duration_norm)
            if trade:
                trades.append(trade)
                self.market_state.last_trade_step[instrument] = self.market_state.current_step
                
                # FIXED: Update risk manager after trade
                self.risk_manager.post_trade_update(trade)
                
        return trades
        
    def _execute_single_trade(
        self,
        instrument: str,
        intensity: float,
        duration_norm: float
    ) -> Optional[Dict[str, Any]]:
        """Execute a single trade (live or simulated)"""
        # Get current market data
        df = self.data[instrument]["D1"]
        if self.market_state.current_step >= len(df):
            return None
            
        bar = df.iloc[self.market_state.current_step]
        price = float(bar["close"])
        
        # Get volatility with safety checks
        vol = self._get_instrument_volatility(instrument)
        
        # Calculate position size
        size = self._calculate_position_size(instrument, intensity, vol)
        
        if size == 0.0:
            self.logger.debug(f"Zero position size for {instrument}")
            return None
            
        # Execute based on mode
        if self.config.live_mode:
            return self._execute_live_trade(instrument, size, intensity)
        else:
            return self._execute_simulated_trade(
                instrument, size, intensity, duration_norm, price, df
            )
            
    def _execute_simulated_trade(
        self,
        instrument: str,
        size: float,
        intensity: float,
        duration_norm: float,
        entry_price: float,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute a simulated trade"""
        side = "BUY" if intensity > 0 else "SELL"
        self.logger.info(
            f"[SIM] {side} {instrument} {size:.3f} lots @ {entry_price:.4f}"
        )
        
        # Calculate holding period
        hold_steps = max(
            int(duration_norm * self.config.max_steps), 
            1
        )
        exit_idx = min(
            self.market_state.current_step + hold_steps,
            len(df) - 1
        )
        
        # Get exit price
        exit_price = float(df.iloc[exit_idx]["close"])
        
        # Calculate PnL
        point_value = self.point_value.get(instrument, 1.0)
        if intensity > 0:
            pnl = (exit_price - entry_price) * size * point_value
        else:
            pnl = (entry_price - exit_price) * size * point_value
            
        # Sanitize PnL
        pnl = np.clip(
            np.nan_to_num(pnl, nan=0.0),
            -10 * self.config.initial_balance,
            10 * self.config.initial_balance
        )
        
        # Update balance
        self.market_state.balance += pnl
        self.market_state.peak_balance = max(
            self.market_state.peak_balance,
            self.market_state.balance
        )
        self.market_state.current_drawdown = max(
            (self.market_state.peak_balance - self.market_state.balance) / 
            (self.market_state.peak_balance + 1e-12),
            0.0
        )
        
        self.logger.info(
            f"[SIM] Trade closed: exit @ {exit_price:.4f}, PnL={pnl:.2f}"
        )
        
        return {
            "instrument": instrument,
            "pnl": pnl,
            "duration": hold_steps,
            "exit_reason": "timeout",
            "size": size if intensity > 0 else -size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "features": np.array([exit_price, pnl, hold_steps], dtype=np.float32),
        }
        
    def _execute_live_trade(
        self,
        instrument: str,
        size: float,
        intensity: float
    ) -> Optional[Dict[str, Any]]:
        """FIXED: Execute a live trade via MetaTrader5"""
        symbol = instrument.replace("/", "")
        
        # Select symbol
        if not mt5.symbol_select(symbol, True):
            self.logger.warning(f"Cannot select symbol {symbol}")
            return None
            
        # Get symbol info
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            self.logger.warning(f"No symbol info for {symbol}")
            return None
            
        # Round size to broker requirements
        size = self._round_lot_size(size, symbol_info)
        
        # Get current tick
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            self.logger.warning(f"No tick data for {symbol}")
            return None
            
        # Determine price based on side
        price = tick.ask if intensity > 0 else tick.bid
        
        # Create order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": size,
            "type": mt5.ORDER_TYPE_BUY if intensity > 0 else mt5.ORDER_TYPE_SELL,
            "price": price,
            "deviation": 20,
            "magic": 202406,
            "comment": f"AI trade ep{self.episode_count}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Send order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.logger.warning(f"Order failed: {result.comment}")
            return None
            
        # Get the actual executed price
        executed_price = result.price if hasattr(result, 'price') else price
        
        # Register position
        self.position_manager.open_positions[instrument] = {
            "ticket": result.order,
            "side": 1 if intensity > 0 else -1,
            "lots": size,
            "price_open": executed_price,
            "peak_profit": 0.0,
            "entry_step": self.market_state.current_step,  # FIXED: Added for monitoring
            "instrument": instrument,  # FIXED: Added for standardization
        }
        
        self.logger.info(
            f"[LIVE] {'BUY' if intensity > 0 else 'SELL'} {instrument} "
            f"{size:.3f} lots @ {executed_price:.4f}, ticket={result.order}"
        )
        
        # FIXED: Return proper trade dictionary
        return {
            "instrument": instrument,
            "size": size,
            "entry_price": executed_price,
            "side": "BUY" if intensity > 0 else "SELL",
            "ticket": result.order,
            "pnl": 0.0,  # Initial PnL is zero
            "duration": 0,
            "exit_reason": "open",
            "features": np.array([executed_price, size, intensity], dtype=np.float32),
        }
        
    def _calculate_position_size(
        self,
        instrument: str,
        intensity: float,
        volatility: float
    ) -> float:
        """Calculate position size with risk management - FIXED for better trading"""
        # Base size from balance - increased allocation
        risk_pct = 0.02  # Risk 2% per trade
        risk_capital = self.market_state.balance * risk_pct
        
        # Adjust for volatility
        vol_adj = min(0.02 / (volatility + 1e-12), 2.0)
        
        # Scale by intensity with minimum size
        base_size = max(
            risk_capital * abs(intensity) * vol_adj / 100000,
            0.01  # Minimum 0.01 lots
        )
        
        # Apply instrument-specific limits
        if "XAU" in instrument or "GOLD" in instrument:
            base_size = np.clip(base_size, 0.01, 1.0)  # 0.01-1.0 lots for gold
        else:
            base_size = np.clip(base_size, 0.01, 5.0)  # 0.01-5.0 lots for forex
            
        return round(base_size, 2)
        
    def _round_lot_size(self, size: float, symbol_info) -> float:
        """Round lot size to broker requirements"""
        if hasattr(symbol_info, 'volume_step'):
            step = symbol_info.volume_step
            return round(size / step) * step
        return round(size, 2)
        
    def _finalize_step(
        self,
        trades: List[Dict],
        actions: np.ndarray,
        consensus: float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Finalize step with reward calculation and state updates"""
        # 1) Store trades
        self.trades = trades
        self.episode_metrics.trades.extend(trades)

        # 2) Calculate step PnL
        step_pnl = sum(t.get("pnl", 0.0) for t in trades)
        self.episode_metrics.pnls.append(step_pnl)

        # --- feed RegimePerformanceMatrix ----------------------------------
        #   a) get realized volatility for this bar
        vol = self._get_current_volatility()

        #   b) ask your FractalRegimeConfirmation what it predicted
        reg_label, _ = self.fractal_confirm.step(
            data_dict=self.data,
            current_step=self.market_state.current_step,
            theme_detector=self.theme_detector
        )
        #   c) map label → index (must match your RPM.n)
        regime_map = {"noise": 0, "volatile": 1, "trending": 2}
        pred_idx = regime_map.get(reg_label, 0)

        #   d) update the matrix
        self.regime_matrix.step(
            pnl=step_pnl,
            volatility=vol,
            predicted_regime=pred_idx
        )
        # --------------------------------------------------------------------

        # 3) Calculate reward
        reward = self._calculate_reward(trades, actions, consensus)

        # 4) Check for termination
        terminated = self._check_termination()

        # 5) Build next observation
        obs = self._get_next_observation(trades, actions)

        # 6) Pack info dict
        info = self._create_step_info(trades, step_pnl, consensus)

        # 7) Update trading mode
        self._update_mode_manager(trades, step_pnl, consensus)

        # 8) Advance step counter
        self.market_state.current_step += 1
        self.current_step = self.market_state.current_step  # keep in sync

        # 9) Handle end of episode
        if terminated:
            self._handle_episode_end(step_pnl)

        # 10) Return exactly the Gym API tuple
        return obs, float(reward), terminated, False, info

        
    def _create_no_trade_step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Create step output when no trades are executed"""
        return self._finalize_step([], actions, 0.0)
        
    # ═══════════════════════════════════════════════════════════════════
    #  Helper Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _get_initial_balance(self) -> float:
        """Get initial balance for episode"""
        if self.config.live_mode:
            info = mt5.account_info()
            if info and hasattr(info, "balance"):
                return float(info.balance)
        return self.config.initial_balance
        
    def _select_starting_step(self) -> int:
        """Select random starting step for episode"""
        # Find minimum available bars across instruments
        min_bars = min(
            len(self.data[inst]["D1"])
            for inst in self.instruments
            if "D1" in self.data[inst]
        )
        
        # Ensure we have enough data
        min_start = 100  # Need history for indicators
        max_start = max(min_bars - self.config.max_steps - 1, min_start)
        
        if max_start <= min_start:
            self.logger.warning(
                f"Insufficient data: {min_bars} bars, using step 0"
            )
            return 0
            
        return random.randint(min_start, max_start)
        
    def _reset_all_modules(self):
        """Reset all modules to initial state"""
        # Reset pipeline modules
        self.pipeline.reset()
        
        # Reset other modules
        modules_to_reset = [
            self.reward_shaper, self.meta_rl, self.arbiter,
            self.consensus, self.collusion, self.haligner,
            self.strategy_pool, self.meta_agent, self.meta_planner,
            self.long_term_memory, self.world_model
        ]
        
        for module in modules_to_reset:
            if hasattr(module, 'reset'):
                try:
                    module.reset()
                except Exception as e:
                    self.logger.warning(f"Failed to reset {module.__class__.__name__}: {e}")
                    
    def _prime_risk_system(self):
        """Prime risk system with historical price data"""
        try:
            price_dict = {}
            for inst in self.instruments:
                df = self.data[inst]["D1"]
                start_idx = max(0, self.market_state.current_step - self.risk_system.var_window)
                end_idx = self.market_state.current_step + 1
                prices = df["close"].iloc[start_idx:end_idx].values
                price_dict[inst] = prices
                
            if all(len(p) > 10 for p in price_dict.values()):
                self.risk_system.prime_returns_with_history(price_dict)
            else:
                self.risk_system.prime_returns_with_random()
        except Exception as e:
            self.logger.warning(f"Failed to prime risk system: {e}")
            self.risk_system.prime_returns_with_random()
            
    def _select_strategy_genome(self):
        """Select strategy genome for episode"""
        self.strategy_pool.select_genome("random")
        self.current_genome = self.strategy_pool.active_genome.copy()
        self.logger.info(f"Selected genome: {self.current_genome}")
        
    def _sanitize_observation(self, obs: np.ndarray) -> np.ndarray:
        """Ensure observation contains no invalid values"""
        # Replace NaN/Inf with zeros
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values
        obs = np.clip(obs, -1e6, 1e6)
        
        # Validate
        assert np.all(np.isfinite(obs)), "Non-finite values in observation"
        
        return obs
        
    def _get_current_volatility(self) -> float:
        """Get current volatility for primary instrument"""
        try:
            df = self.data[self.instruments[0]]["D1"]
            if self.market_state.current_step < len(df):
                return float(df.iloc[self.market_state.current_step]["volatility"])
        except Exception:
            pass
        return 0.01  # Safe default
        
    def _get_instrument_volatility(self, instrument: str) -> float:
        """Get volatility for specific instrument"""
        try:
            df = self.data[instrument]["D1"]
            if self.market_state.current_step < len(df):
                vol = df.iloc[self.market_state.current_step].get("volatility", 0.01)
                vol = float(np.nan_to_num(vol, nan=0.01))
                return max(vol, self.position_manager.min_volatility)
        except Exception:
            pass
        return self.position_manager.min_volatility
        
    def _get_price_history(self, instrument: str, timeframe: str) -> np.ndarray:
        """Get recent price history for instrument/timeframe"""
        try:
            df = self.data[instrument][timeframe]
            end_idx = min(self.market_state.current_step + 1, len(df))
            start_idx = max(0, end_idx - 7)
            
            if end_idx > start_idx:
                prices = df["close"].iloc[start_idx:end_idx].values
                # Pad if needed
                if len(prices) < 7:
                    prices = np.pad(prices, (7 - len(prices), 0), mode='edge')
                return prices[-7:].astype(np.float32)
        except Exception as e:
            self.logger.warning(f"Failed to get price history: {e}")
            
        return np.zeros(7, dtype=np.float32)
        
    def _get_recent_returns(self) -> Dict[str, np.ndarray]:
        """Get recent returns for risk calculations"""
        returns = {}
        for inst in self.instruments:
            try:
                df = self.data[inst]["D1"]
                end_idx = self.market_state.current_step + 1
                start_idx = max(0, end_idx - 20)
                
                if end_idx > start_idx + 1:
                    prices = df["close"].iloc[start_idx:end_idx].values
                    ret = np.diff(np.log(prices))
                    returns[inst] = ret
                else:
                    returns[inst] = np.array([])
            except Exception:
                returns[inst] = np.array([])
                
        return returns
        
    def get_instrument_correlations(self) -> Dict[str, float]:
        """Calculate pairwise correlations between instruments"""
        correlations = {}
        
        try:
            # Get returns for all instruments
            returns_dict = self._get_recent_returns()
            
            # Calculate pairwise correlations
            for i, inst1 in enumerate(self.instruments):
                for j, inst2 in enumerate(self.instruments):
                    if i >= j:
                        continue
                        
                    ret1 = returns_dict.get(inst1, np.array([]))
                    ret2 = returns_dict.get(inst2, np.array([]))
                    
                    if len(ret1) > 5 and len(ret2) > 5:
                        # Align lengths
                        min_len = min(len(ret1), len(ret2))
                        ret1 = ret1[-min_len:]
                        ret2 = ret2[-min_len:]
                        
                        # Calculate correlation
                        corr = np.corrcoef(ret1, ret2)[0, 1]
                        correlations[f"{inst1}-{inst2}"] = float(np.nan_to_num(corr))
                    else:
                        correlations[f"{inst1}-{inst2}"] = 0.0
                        
        except Exception as e:
            self.logger.warning(f"Failed to calculate correlations: {e}")
            
        return correlations
        
    def _calculate_reward(
        self,
        trades: List[Dict],
        actions: np.ndarray,
        consensus: float
    ) -> float:
        """Calculate step reward"""
        # Get reward from reward shaper
        reward = self.reward_shaper.shape_reward(
            trades=trades,
            balance=self.market_state.balance,
            drawdown=self.market_state.current_drawdown,
            consensus=consensus,
            actions=actions,
        )
        
        # Store for next step
        self._last_reward = reward
        
        return reward
        
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Max steps reached
        if self.market_state.current_step >= self.config.max_steps - 1:
            return True
            
        # Data exhausted
        for inst in self.instruments:
            if self.market_state.current_step >= len(self.data[inst]["D1"]) - 1:
                return True
                
        # Catastrophic loss
        if self.market_state.balance < self.config.initial_balance * 0.5:
            self.logger.warning("Episode terminated: 50% loss")
            return True
            
        # Extreme drawdown
        if self.market_state.current_drawdown > 0.5:
            self.logger.warning("Episode terminated: 50% drawdown")
            return True
            
        return False
    

    @profile_method
    def _get_full_observation(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Build a unified observation by passing `standardized_data` through
        the pipeline, then guard against NaNs before sanitizing/caching.
        """
        cache_key = (self.market_state.current_step, id(data))
        if cache_key in self._obs_cache:
            return self._obs_cache[cache_key]

        # augment inputs for modules
        standardized_data = data.copy()
        standardized_data.update({
            "env": self,
            "open_positions": list(self.position_manager.open_positions.values()),
            "current_step": self.market_state.current_step,
            "data_dict": self.data,
            "price": standardized_data.get("price", None),
            "balance": self.market_state.balance,
            "drawdown": self.market_state.current_drawdown,
            "instruments": self.instruments,
        })

        obs = self.pipeline.step(standardized_data)

        # ←— New sanity check right after pipeline
        if not np.all(np.isfinite(obs)):
            self.logger.error(f"🛑 NaN/Inf in obs at step {self.market_state.current_step}: {obs}")
            self.logger.error(f"  input keys: {list(standardized_data.keys())}")
            for k, v in standardized_data.items():
                if isinstance(v, np.ndarray) and not np.all(np.isfinite(v)):
                    self.logger.error(f"    '{k}' has bad values: {v}")
            import pdb; pdb.set_trace()
        # ————————————————————————————————————————

        # ensure matches declared observation_space size
        if hasattr(self, "observation_space"):
            target = self.observation_space.shape[0]
            if obs.size < target:
                obs = np.concatenate([obs, np.zeros(target - obs.size, dtype=np.float32)])
            elif obs.size > target:
                obs = obs[:target]

        # sanitize infinities, extreme outliers, and cache
        obs = self._sanitize_observation(obs)
        self._obs_cache[cache_key] = obs

        # trim cache if it grows too big
        if len(self._obs_cache) > 100:
            for key in list(self._obs_cache)[:-100]:
                del self._obs_cache[key]

        return obs


    def _get_next_observation(
        self,
        trades: List[Dict],
        actions: np.ndarray
    ) -> np.ndarray:
        """Get observation for next step"""
        # Get price histories
        hist_h1 = self._get_price_history(self.instruments[0], "H1")
        hist_h4 = self._get_price_history(self.instruments[0], "H4")
        hist_d1 = self._get_price_history(self.instruments[0], "D1")
        
        # Calculate PnL
        pnl = sum(t.get("pnl", 0.0) for t in trades)
        
        # Get memory embedding
        memory = (
            self.feature_engine.last_embedding
            if hasattr(self.feature_engine, 'last_embedding')
            else None
        )
        
        # FIXED: Create observation data with standardized naming
        obs_data = {
            "env": self,
            "price_h1": hist_h1,
            "price_h4": hist_h4,
            "price_d1": hist_d1,
            "actions": actions,
            "trades": trades,
            "open_positions": list(self.position_manager.open_positions.values()),  # FIXED
            "drawdown": self.market_state.current_drawdown,
            "memory": memory,
            "pnl": pnl,
            "correlations": self.get_instrument_correlations(),
            "current_step": self.market_state.current_step,  # FIXED: Added
        }
        
        # Get full observation
        obs = self._get_full_observation(obs_data)
        
        # Sanitize
        obs = self._sanitize_observation(obs)
        
        # Update meta-RL
        if obs.size == self.meta_rl.obs_dim:
            self.meta_rl.record_step(obs, self._last_reward if hasattr(self, '_last_reward') else 0.0)
            
        # Store for next step
        self._last_actions = actions.copy()
        self._last_reward = self._last_reward if hasattr(self, '_last_reward') else 0.0
        
        return obs
        
    def _create_reset_info(self) -> Dict[str, Any]:
        """Create info dict for reset"""
        return {
            "episode": self.episode_count,
            "balance": self.market_state.balance,
            "genome": self.current_genome,
            "start_step": self.market_state.current_step,
        }
        
    def _create_step_info(
        self,
        trades: List[Dict],
        pnl: float,
        consensus: float
    ) -> Dict[str, Any]:
        """Create info dict for step"""
        return {
            "balance": self.market_state.balance,
            "pnl": pnl,
            "drawdown": self.market_state.current_drawdown,
            "trades": len(trades),
            "consensus": consensus,
            "mode": self.mode_manager.get_mode(),
            "step": self.market_state.current_step,
            "positions": len(self.position_manager.open_positions),
        }
        
    def _update_mode_manager(
        self,
        trades: List[Dict],
        pnl: float,
        consensus: float
    ):
        """Update trading mode based on performance"""
        self.mode_manager.update(
            pnl=pnl,
            drawdown=self.market_state.current_drawdown,
            consensus=consensus,
            trade_count=len(trades),
        )
        
    def _handle_episode_end(self, final_pnl: float):
        """Handle episode termination"""
        # Log episode summary
        total_pnl = sum(self.episode_metrics.pnls)
        total_trades = len(self.episode_metrics.trades)
        max_dd = max(self.episode_metrics.drawdowns) if self.episode_metrics.drawdowns else 0
        
        self.logger.info(
            f"Episode {self.episode_count} ended: "
            f"PnL={total_pnl:.2f}, Trades={total_trades}, MaxDD={max_dd:.2%}"
        )
        
        # Save checkpoints
        if self.episode_count % 100 == 0:
            try:
                self._save_checkpoints()
            except Exception as e:
                self.logger.error(f"Failed to save checkpoints: {e}")
                
    def set_module_enabled(self, name: str, enabled: bool):
        """Enable or disable a module"""
        if name not in self.module_enabled:
            raise KeyError(f"Unknown module: {name}")
        self.module_enabled[name] = enabled
        self.logger.info(f"Module {name} {'enabled' if enabled else 'disabled'}")
        
    def get_state(self) -> Dict[str, Any]:
        """Get complete environment state for serialization"""
        return {
            "market_state": {
                "balance": self.market_state.balance,
                "peak_balance": self.market_state.peak_balance,
                "current_step": self.market_state.current_step,
                "current_drawdown": self.market_state.current_drawdown,
                "last_trade_step": self.market_state.last_trade_step,
            },
            "episode_metrics": {
                "pnls": self.episode_metrics.pnls,
                "durations": self.episode_metrics.durations,
                "drawdowns": self.episode_metrics.drawdowns,
                "trades": self.episode_metrics.trades,
                "votes_log": self.episode_metrics.votes_log,
                "reasoning_trace": self.episode_metrics.reasoning_trace,
            },
            "episode_count": self.episode_count,
            "position_manager": self.position_manager.get_state(),
        }
        
    def set_state(self, state: Dict[str, Any]):
        """Restore environment state from serialization"""
        # Restore market state
        ms = state.get("market_state", {})
        self.market_state.balance = ms.get("balance", self.config.initial_balance)
        self.market_state.peak_balance = ms.get("peak_balance", self.market_state.balance)
        self.market_state.current_step = ms.get("current_step", 0)
        self.market_state.current_drawdown = ms.get("current_drawdown", 0.0)
        self.market_state.last_trade_step = ms.get("last_trade_step", {})
        
        # Restore episode metrics
        em = state.get("episode_metrics", {})
        self.episode_metrics.pnls = em.get("pnls", [])
        self.episode_metrics.durations = em.get("durations", [])
        self.episode_metrics.drawdowns = em.get("drawdowns", [])
        self.episode_metrics.trades = em.get("trades", [])
        self.episode_metrics.votes_log = em.get("votes_log", [])
        self.episode_metrics.reasoning_trace = em.get("reasoning_trace", [])
        
        # Restore other state
        self.episode_count = state.get("episode_count", 0)
        self.current_step = self.market_state.current_step  # FIXED: Sync step counters
        
        # Restore position manager
        if "position_manager" in state:
            self.position_manager.set_state(state["position_manager"])
            
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """Render the environment"""
        if mode == "human":
            # Text output
            print(
                f"Step {self.market_state.current_step} | "
                f"Mode: {self.mode_manager.get_mode().upper()} | "
                f"Balance: ${self.market_state.balance:.2f} | "
                f"Drawdown: {self.market_state.current_drawdown:.2%} | "
                f"Trades: {len(self.episode_metrics.trades)}"
            )
        elif mode == "rgb_array":
            # Could implement visual rendering here
            return None
            
        return None
        
    def close(self):
        """Clean up resources"""
        # Save final checkpoints
        try:
            self._save_checkpoints()
        except Exception as e:
            self.logger.error(f"Failed to save final checkpoints: {e}")
            
        # Close loggers
        for handler in self.logger.handlers:
            handler.close()
            
        self.logger.info("Environment closed")
        
    # ═══════════════════════════════════════════════════════════════════
    #  Checkpointing Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def _save_checkpoints(self):
        """Save all module checkpoints"""
        try:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            
            # Save environment state
            state_path = os.path.join(
                self.config.checkpoint_dir,
                f"env_state_ep{self.episode_count}.pkl"
            )
            with open(state_path, "wb") as f:
                pickle.dump(self.get_state(), f)
                
            # Save module states
            modules_to_save = [
                self.position_manager,
                self.risk_controller,
                self.risk_system,
                self.strategy_pool,
                self.mistake_memory,
                self.meta_rl,
            ]
            
            for module in modules_to_save:
                if hasattr(module, 'save_checkpoint'):
                    try:
                        module.save_checkpoint(
                            os.path.join(
                                self.config.checkpoint_dir,
                                f"{module.__class__.__name__}_ep{self.episode_count}.pkl"
                            )
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to save {module.__class__.__name__}: {e}"
                        )
                        
            self.logger.info(f"Saved checkpoints for episode {self.episode_count}")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")


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