     # Update balance from# envs/ppo_env.py
"""
State-of-the-art PPO Trading Environment
Fully refactored for robustness, clarity, and module compatibility
"""
from __future__ import annotations

# ─────────────────────────── Std-lib ──────────────────────────────────
from itertools import combinations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict
import math
import os
import copy
import random
import logging
import pickle
import warnings

# ───────────────────────── Third-party ────────────────────────────────
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces
import MetaTrader5 as mt5

# ──────────────────────── Internal modules ────────────────────────────
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
    
    # Trading parameters
    no_trade_penalty: float = 0.3
    consensus_min: float = 0.30
    consensus_max: float = 0.70
    max_episodes: int = 10000
    
    # Risk parameters
    min_intensity: float = 0.25
    min_inst_confidence: float = 0.60
    rotation_gap: int = 5
    
    # Module flags
    live_mode: bool = True
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False
    
    # Logging
    log_dir: str = "logs"
    log_level: str = "INFO"


@dataclass
class MarketState:
    """Encapsulates current market state"""
    balance: float
    peak_balance: float
    current_step: int
    current_drawdown: float
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
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
# ║                     Processing Pipeline Wrapper                      ║
# ╚═════════════════════════════════════════════════════════════════════╝

class TradingPipeline:
    """Manages the sequential processing of trading modules"""
    
    def __init__(self, modules: List[Module]):
        self.modules = modules
        self._module_map = {m.__class__.__name__: m for m in modules}
        
    def reset(self):
        """Reset all modules in the pipeline"""
        for module in self.modules:
            try:
                module.reset()
            except Exception as e:
                logging.warning(f"Failed to reset {module.__class__.__name__}: {e}")
                
    def step(self, data: Dict[str, Any]) -> np.ndarray:
        """Process data through all enabled modules"""
        env = data.get("env")
        observations = []
        
        for module in self.modules:
            if env and not env.module_enabled.get(module.__class__.__name__, True):
                continue
                
            try:
                # Get method signature and call with matching parameters
                sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
                kwargs = {k: data[k] for k in sig if k in data}
                module.step(**kwargs)
                
                # Collect observation components
                obs = module.get_observation_components()
                observations.append(obs)
            except Exception as e:
                logging.error(f"Error in {module.__class__.__name__}.step(): {e}")
                # Append zeros on error to maintain shape consistency
                observations.append(np.zeros(0, dtype=np.float32))
                
        # Concatenate all observations
        return np.concatenate(observations) if observations else np.zeros(0, dtype=np.float32)
    
    def get_module(self, name: str) -> Optional[Module]:
        """Get a module by class name"""
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
        
        # Use provided config or create default
        self.config = config or TradingConfig()
        
        # Initialize logging
        self._setup_logging()
        
        # Set random seeds for reproducibility
        self._set_seeds(self.config.init_seed)
        
        # Initialize market state
        initial_bal = self._get_initial_balance()
        self.market_state = MarketState(
            balance=initial_bal,
            peak_balance=initial_bal,
            current_step=0,
            current_drawdown=0.0,
        )
        
        # Initialize episode tracking
        self.episode_metrics = EpisodeMetrics()
        self.episode_count = 0
        
        # Store data
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(data_dict.keys())
        
        # Validate data
        self._validate_data()
        
        # Initialize all modules
        self._initialize_modules()
        
        # Setup observation and action spaces
        self._setup_spaces()
        
        # Initialize module states
        self._initialize_module_states()
        
        # Setup checkpointing
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        self._maybe_load_checkpoints()
        
        self.logger.info("Environment initialized successfully")
        
    def _setup_logging(self):
        """Configure logging for the environment"""
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.FileHandler(
                os.path.join(self.config.log_dir, "env.log")
            )
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            
        self.logger.setLevel(getattr(logging, self.config.log_level))
        self.logger.propagate = False
        
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def _validate_data(self):
        """Validate input data structure"""
        for inst in self.instruments:
            if inst not in self.data:
                raise ValueError(f"Missing instrument: {inst}")
            for tf in ["H1", "H4", "D1"]:
                if tf not in self.data[inst]:
                    raise ValueError(f"Missing timeframe {tf} for {inst}")
                df = self.data[inst][tf]
                required_cols = ["open", "high", "low", "close", "volume", "volatility"]
                missing = set(required_cols) - set(df.columns)
                if missing:
                    raise ValueError(f"Missing columns {missing} in {inst}/{tf}")
                    
    def _initialize_modules(self):
        """Initialize all trading modules with proper configuration"""
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
        self.explainer = ExplanationGenerator(debug=self.config.debug)
        
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
        
        # Create module pipeline
        self._create_pipeline()
        
    def _create_pipeline(self):
        """Create the processing pipeline with all modules"""
        core_modules = [
            self.feature_engine, self.compliance, self.risk_system,
            self.theme_detector, self.time_risk_scaler, self.liquidity_layer,
            self.strategy_intros, self.curriculum_planner, self.memory_budget,
            self.bias_auditor, self.opp_enhancer, self.thesis_engine,
            self.regime_matrix, self.trade_thesis,
            self.mode_manager, self.active_monitor, self.corr_controller,
            self.dd_rescue, self.exec_monitor, self.anomaly_detector,
            self.position_manager, self.fractal_confirm, self.regime_switcher,
        ]
        
        # Add simulation modules if not in live mode
        if not self.config.live_mode and self.config.enable_shadow_sim:
            core_modules.extend([
                self.shadow_sim, self.role_coach, self.opponent_sim
            ])
            
        self.pipeline = TradingPipeline(core_modules)
        
        # Module enable/disable tracking
        self.module_enabled = {
            m.__class__.__name__: True for m in core_modules
        }
        
    def _setup_spaces(self):
        """Setup observation and action spaces"""
        # 1) Compute and store action_dim up front so dummy builder can use it
        self.action_dim = 2 * len(self.instruments)  # intensity and duration per instrument

        # 2) Build a dummy input and get its full observation to size the obs space
        dummy_obs = self._get_full_observation(self._create_dummy_input())
        obs_size = dummy_obs.shape[0]

        # 3) Define the observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

        # 4) Define the action space using our precomputed action_dim
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )
        self.action_space.seed(self.config.init_seed)

        # 5) Let any modules that care know the action_dim
        for module in [
            self.regime_switcher,
            self.theme_detector,
            self.fractal_confirm,
            self.liquidity_layer,
            self.risk_controller
        ]:
            if hasattr(module, 'set_action_dim'):
                module.set_action_dim(self.action_dim)

                
    def _initialize_module_states(self):
        """Initialize module states that depend on spaces being defined"""
        # Initialize MetaRL controller
        self.meta_rl = MetaRLController(
            self.observation_space.shape[0],
            self.action_space.shape[0]
        )
        
        # Create expert wrappers for voting
        self._create_voting_committee()
        
        # Initialize strategy arbiter
        self._initialize_arbiter()
        
        # Initialize consensus systems
        self.consensus = ConsensusDetector(len(self.committee), 0.7)
        self.collusion = CollusionAuditor(len(self.committee), 0.95)
        self.haligner = TimeHorizonAligner([0] * len(self.committee))
        self.alt_sampler = AlternativeRealitySampler(len(self.committee))
        
    def _create_voting_committee(self):
        """Create the voting committee with expert wrappers"""
        theme_expert = ThemeExpert(self.theme_detector, self)
        season_expert = SeasonalityRiskExpert(self.time_risk_scaler, self)
        meta_rl_expert = MetaRLExpert(self.meta_rl, self)
        veto_expert = TradeMonitorVetoExpert(self.active_monitor, self)
        regime_expert = RegimeBiasExpert(self.fractal_confirm, self)
        
        self.committee = [
            self.position_manager,
            self.risk_controller,
            self.liquidity_layer,
            theme_expert,
            season_expert,
            meta_rl_expert,
            veto_expert,
            regime_expert,
        ]
        
    def _initialize_arbiter(self):
        """Initialize the strategy arbiter"""
        arbiter_members = [
            self.liquidity_layer,
            self.fractal_confirm,
            self.theme_detector,
            self.regime_switcher
        ]
        
        self.arbiter = StrategyArbiter(
            members=arbiter_members,
            init_weights=[0.25, 0.25, 0.25, 0.25],
            action_dim=self.action_dim,
            consensus=ConsensusDetector(4),
            horizon_aligner=TimeHorizonAligner([1, 4, 24, 24]),
            debug=self.config.debug,
        )
        
    def _create_dummy_input(self) -> Dict[str, Any]:
        """Create dummy input for initialization"""
        return {
            "env": self,
            "price_h1": np.zeros(7, dtype=np.float32),
            "price_h4": np.zeros(7, dtype=np.float32),
            "price_d1": np.zeros(7, dtype=np.float32),
            "actions": np.zeros(self.action_dim, dtype=np.float32),
            "trades": [],
            "open_trades": [],
            "drawdown": 0.0,
            "memory": np.zeros(32, dtype=np.float32),
            "pnl": 0.0,
            "correlations": {},
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
            
    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Validate and sanitize action array"""
        # Ensure numpy array
        actions = np.asarray(actions, dtype=np.float32)
        
        # Check shape
        if actions.shape != self.action_space.shape:
            raise ValueError(f"Invalid action shape: {actions.shape}")
            
        # Check for NaN/Inf
        if not np.all(np.isfinite(actions)):
            self.logger.warning("Non-finite values in actions, replacing with zeros")
            actions = np.nan_to_num(actions, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Clip to valid range
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        
        return actions
        
    def _apply_meta_rl(self, actions: np.ndarray) -> np.ndarray:
        """Apply meta-RL overlay to actions"""
        try:
            # Get current observation for meta-RL
            obs = self._get_full_observation(self._create_dummy_input())
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            # Get meta-RL action adjustment
            meta_output = self.meta_rl.act(obs_tensor)
            meta_action = (
                meta_output["action"] if isinstance(meta_output, dict) 
                else meta_output
            )
            meta_action = np.asarray(meta_action).reshape(-1)
            
            # Validate meta action
            if not np.all(np.isfinite(meta_action)):
                self.logger.warning("Non-finite meta-RL action, using zeros")
                meta_action = np.zeros_like(meta_action)
                
            # Apply genome-scaled adjustment
            genome_scale = self.strategy_pool.active_genome[2]
            adjusted = actions + meta_action * genome_scale
            
            # Clip to valid range
            return np.clip(adjusted, self.action_space.low, self.action_space.high)
            
        except Exception as e:
            self.logger.error(f"Meta-RL application failed: {e}")
            return actions
            
    def _pass_risk_checks(self) -> bool:
        """Perform pre-trade risk checks"""
        # Get current volatility
        vol = self._get_current_volatility()
        
        # Update arbiter market state
        self.arbiter.update_market_state(vol)
        
        # Check correlations
        corr_dict = self.get_instrument_correlations()
        if self.corr_controller.step(correlations=corr_dict):
            self.position_manager.max_pct *= 0.5
            self.logger.info(
                f"High correlation detected, reducing position size to {self.position_manager.max_pct:.4f}"
            )
            
        # Check drawdown rescue
        if self.dd_rescue.step(current_drawdown=self.market_state.current_drawdown):
            self.logger.info("Drawdown rescue triggered, skipping trades")
            return False
            
        # Adjust dynamic risk
        self.risk_controller.adjust_risk({
            "drawdown": self.market_state.current_drawdown,
            "volatility": vol
        })
        
        # Apply risk coefficient to position sizing
        risk_coef = float(self.risk_controller.get_observation_components()[0])
        base_pct = getattr(
            self.position_manager, 
            "default_max_pct", 
            self.position_manager.max_pct
        )
        self.position_manager.max_pct = min(base_pct * risk_coef, base_pct)
        
        return True
        
    def _get_committee_decision(self, actions: np.ndarray) -> np.ndarray:
        """Get blended decision from voting committee"""
        votes_by_sym_tf = {}
        blended_by_sym_tf = {}
        committee_names = [m.__class__.__name__ for m in self.arbiter.members]
        
        # Collect votes for each instrument/timeframe combination
        for inst in self.instruments:
            for tf in ["H1", "H4", "D1"]:
                # Get price history
                hist = self._get_price_history(inst, tf)
                
                # Create observation for this timeframe
                obs_data = {
                    "env": self,
                    "price_h1": hist if tf == "H1" else np.zeros_like(hist),
                    "price_h4": hist if tf == "H4" else np.zeros_like(hist),
                    "price_d1": hist if tf == "D1" else np.zeros_like(hist),
                    "actions": actions,
                }
                obs = self._get_full_observation(obs_data)
                
                # Get arbiter's blended proposal
                blend = self.arbiter.propose(obs)
                
                # Store votes
                if self.arbiter.last_alpha is not None:
                    alpha = self.arbiter.last_alpha.copy()
                else:
                    alpha = np.zeros(self.action_dim)
                    
                votes_by_sym_tf[(inst, tf)] = dict(zip(committee_names, alpha.tolist()))
                blended_by_sym_tf[(inst, tf)] = blend
                
        # Store votes for logging
        self.episode_metrics.votes_log.append(votes_by_sym_tf)
        
        # Blend across timeframes
        return self._blend_committee_votes(blended_by_sym_tf)
        
    def _blend_committee_votes(self, blended_by_sym_tf: Dict[Tuple[str, str], np.ndarray]) -> np.ndarray:
        """Average the per-timeframe blends into final actions"""
        final_actions = np.zeros(self.action_dim, dtype=np.float32)
        
        for i, inst in enumerate(self.instruments):
            timeframe_actions = []
            
            # Collect actions for this instrument across timeframes
            for tf in ["H1", "H4", "D1"]:
                key = (inst, tf)
                if key in blended_by_sym_tf:
                    # Extract instrument-specific actions
                    inst_actions = blended_by_sym_tf[key][2*i:2*i+2]
                    timeframe_actions.append(inst_actions)
                    
            # Average across timeframes if we have data
            if timeframe_actions:
                avg_actions = np.mean(timeframe_actions, axis=0)
                final_actions[2*i:2*i+2] = avg_actions
                
        # Ensure no NaN values
        final_actions = np.nan_to_num(final_actions, nan=0.0)
        
        return final_actions
        
    def _calculate_consensus(self) -> float:
        """Calculate consensus score from arbiter's alpha values"""
        if self.arbiter.last_alpha is None:
            return 0.0
            
        # Use top-2 average for consensus
        alpha_sorted = np.sort(self.arbiter.last_alpha)
        consensus = float(alpha_sorted[-2:].mean())
        
        return consensus
        
    def _pass_consensus_check(self, consensus: float) -> bool:
        """Check if consensus meets dynamic threshold"""
        # Calculate dynamic threshold based on episode progress
        progress = min(self.episode_count / self.config.max_episodes, 1.0)
        threshold = self.config.consensus_min + (
            self.config.consensus_max - self.config.consensus_min
        ) * progress
        
        if consensus < threshold:
            self.logger.info(
                f"Consensus {consensus:.2f} below threshold {threshold:.2f}, skipping trades"
            )
            return False
            
        return True
        
    # ──────────────────────────────────────────────────────────────────
    #  Execute trades based on the final blended action vector
    # ──────────────────────────────────────────────────────────────────
    def _execute_trades(self, actions: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convert the (intensity, duration_norm) pairs in *actions* into live or
        simulated trades.

        Returns
        -------
        List[Dict[str, Any]]
            A (possibly empty) list of trade dictionaries – never ``None``.
        """
        trades: List[Dict[str, Any]] = []

        # 1) Context snapshot – regime, vol, correlations
        regime_info = self._get_regime_info()

        # 2) Narrative explanation (no side-effects on trading logic)
        self._generate_explanation(actions, regime_info)

        # 3) Diagnostics / gate status for the log
        self._log_gate_diagnostics()

        # 4) Loop over instruments and create trades where permitted
        for i, inst in enumerate(self.instruments):

            # 4-a  Rotation gap check
            last_step = self.market_state.last_trade_step.get(inst, -999)
            if self.market_state.current_step - last_step < self.config.rotation_gap:
                self.logger.debug(f"Skipping {inst}: rotation gap")
                continue

            # 4-b  Extract action components
            intensity      = float(actions[2 * i])
            duration_norm  = float(actions[2 * i + 1])

            # 4-c  Minimum intensity gate
            if abs(intensity) < self.config.min_intensity:
                self.logger.debug(
                    f"Skipping {inst}: |intensity| {intensity:.3f} "
                    f"< {self.config.min_intensity}"
                )
                continue

            # 4-d  Position sizing & execution
            trade = self._execute_single_trade(inst, intensity, duration_norm)
            if trade:
                trades.append(trade)
                # Record last trade step for rotation gating
                self.market_state.last_trade_step[inst] = self.market_state.current_step

        # 5) (Optional) keep the most-recent votes for external inspection
        self._last_votes = self.episode_metrics.votes_log[-1] if self.episode_metrics.votes_log else {}

        # 6) Always return a list – even if empty – so callers can iterate/extend
        return trades

                


    def _generate_explanation(
        self,
        actions: np.ndarray,
        regime_info: Dict[str, Any],
    ):
        """
        Build a narrative of the current decision and hand it to
        ``self.explainer``.  The interface is now consistent with the updated
        ExplanationGenerator (no ``committee=`` kwarg).
        """
        # Names and weights come from the *arbiter*, not the committee
        member_names = [m.__class__.__name__ for m in self.arbiter.members]

        if self.arbiter.last_alpha is not None:
            arbiter_weights = self.arbiter.last_alpha.copy()
        else:
            arbiter_weights = np.zeros(len(member_names), dtype=np.float32)

        # The most recent per-(instrument, timeframe) votes
        votes = getattr(self, "_last_votes", {})

        self.explainer.step(
            actions         = actions,
            arbiter_weights = arbiter_weights,
            member_names    = member_names,
            votes           = votes,
            regime          = regime_info["label"],
            volatility      = regime_info["volatility"],
            drawdown        = self.market_state.current_drawdown,
            genome_metrics  = self.get_genome_metrics(),
        )

        
    def _log_gate_diagnostics(self):
        """Log diagnostic information about trading gates"""
        self.logger.info(
            f"[GATES] Step {self.market_state.current_step} | "
            f"Rotation={self.config.rotation_gap} | "
            f"MinIntensity={self.config.min_intensity} | "
            f"MinConfidence={self.config.min_inst_confidence} | "
            f"MaxPct={self.position_manager.max_pct:.4f}"
        )
        
    def _update_memory_systems(self, trades: List[Dict[str, Any]]):
        """Update all memory systems with trade data"""
        # Update mistake memory
        self.mistake_memory.step(trades=trades)
        
        # Update memory compressor
        if trades:
            self.memory_compressor.compress(self.market_state.current_step, trades)
        else:
            # Use last embedding if no trades
            emb = getattr(self.feature_engine, "last_embedding", None)
            if emb is None:
                emb = np.zeros_like(self.memory_compressor.intuition_vector)
            self.memory_compressor.compress(
                self.market_state.current_step,
                [{"features": emb}]
            )
            
        # Update playbook clusterer
        obs = self._get_full_observation(self._create_dummy_input())
        self.playbook_clusterer.step(trades=trades, obs=obs)
        
    def _calculate_reward(
        self,
        trades: List[Dict[str, Any]],
        actions: np.ndarray,
        pnl: float
    ) -> float:
        """Calculate step reward using risk-adjusted shaper"""
        # Get regime vector
        regime_strength = self.fractal_confirm.regime_strength
        regime_onehot = np.full_like(
            self.reward_shaper.regime_weights,
            regime_strength,
            dtype=np.float32
        )
        
        # Calculate base reward
        reward = self.reward_shaper.step(
            balance=self.market_state.balance,
            trades=trades,
            drawdown=self.market_state.current_drawdown,
            regime_onehot=regime_onehot,
            actions=actions
        )
        
        # Add replay bonus if applicable
        reward += self.replay_analyzer.maybe_replay(self.market_state.current_step)
        
        # Ensure finite
        reward = float(np.nan_to_num(reward, nan=0.0))
        
        return reward
        
    def _update_meta_systems(self, pnl: float, trades: List[Dict[str, Any]]):
        """Update meta-learning and evolution systems"""
        # Update meta agent
        self.meta_agent.record(pnl)
        
        # Update meta planner
        self.meta_planner.record_episode({
            "pnl": pnl,
            "drawdown": self.market_state.current_drawdown
        })
        
        # Update opponent enhancer
        self.opp_enhancer.step(trades=trades, pnl=pnl)
        
        # Update curriculum planner
        obs = self._get_full_observation(self._create_dummy_input())
        self.curriculum_planner.step(result={"pnl": pnl, "obs": obs})
        
    def _run_monitoring_systems(self, trades: List[Dict[str, Any]], pnl: float):
        """Run post-trade monitoring systems"""
        # Execution quality monitor
        self.exec_monitor.step(trade_executions=trades)
        
        # Anomaly detection
        obs = self._get_full_observation(self._create_dummy_input())
        self.anomaly_detector.step(pnl=pnl, obs=obs)
        
        # Shadow simulation (backtest only)
        if not self.config.live_mode and hasattr(self, 'shadow_sim'):
            shadow_trades = self.shadow_sim.simulate(
                env=self,
                actions=self._last_actions if hasattr(self, '_last_actions') else None
            )
            if shadow_trades:
                self.logger.info(f"Shadow sim: {len(shadow_trades)} trades")
                
    def _get_next_observation(
        self,
        actions: np.ndarray,
        trades: List[Dict[str, Any]],
        pnl: float
    ) -> np.ndarray:
        """Construct observation for next step"""
        # Get price histories
        hist_h1 = self._get_price_history(self.instruments[0], "H1")
        hist_h4 = self._get_price_history(self.instruments[0], "H4")
        hist_d1 = self._get_price_history(self.instruments[0], "D1")
        
        # Get memory vector
        memory = self.long_term_memory.retrieve(
            self.feature_engine.last_embedding
            if hasattr(self.feature_engine, 'last_embedding')
            else None
        )
        
        # Create observation data
        obs_data = {
            "env": self,
            "price_h1": hist_h1,
            "price_h4": hist_h4,
            "price_d1": hist_d1,
            "actions": actions,
            "trades": trades,
            "open_trades": list(self.position_manager.open_positions.values()),
            "drawdown": self.market_state.current_drawdown,
            "memory": memory,
            "pnl": pnl,
            "correlations": self.get_instrument_correlations(),
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
                "open_positions": self.market_state.open_positions,
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
        self.market_state.open_positions = ms.get("open_positions", {})
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
            # Save meta-RL state
            if hasattr(self.meta_rl, 'state_dict'):
                torch.save(
                    self.meta_rl.state_dict(),
                    os.path.join(self.config.checkpoint_dir, "meta_rl.pt")
                )
                
            # Save arbiter weights
            np.save(
                os.path.join(self.config.checkpoint_dir, "arbiter_weights.npy"),
                self.arbiter.weights
            )
            
            # Save strategy pool
            with open(os.path.join(self.config.checkpoint_dir, "strategy_pool.pkl"), "wb") as f:
                pickle.dump({
                    "population": self.strategy_pool.population,
                    "fitness": self.strategy_pool.fitness,
                    "epoch": self.strategy_pool.epoch,
                }, f)
                
            # Save module states
            module_states = {}
            for module in self.pipeline.modules:
                if hasattr(module, 'get_state'):
                    module_states[module.__class__.__name__] = module.get_state()
                    
            # Save environment state
            env_state = {
                "state": self.get_state(),
                "module_states": module_states,
                "config": self.config.__dict__,
            }
            
            with open(os.path.join(self.config.checkpoint_dir, "env_state.pkl"), "wb") as f:
                pickle.dump(env_state, f)
                
            self.logger.info("Checkpoints saved successfully")
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
            
    def _maybe_load_checkpoints(self):
        """Load checkpoints if they exist"""
        try:
            # Load meta-RL
            meta_rl_path = os.path.join(self.config.checkpoint_dir, "meta_rl.pt")
            if os.path.exists(meta_rl_path) and hasattr(self.meta_rl, 'load_state_dict'):
                self.meta_rl.load_state_dict(torch.load(meta_rl_path))
                self.logger.info("Loaded meta-RL checkpoint")
                
            # Load arbiter weights
            arbiter_path = os.path.join(self.config.checkpoint_dir, "arbiter_weights.npy")
            if os.path.exists(arbiter_path):
                self.arbiter.weights = np.load(arbiter_path)
                self.logger.info("Loaded arbiter weights")
                
            # Load strategy pool
            pool_path = os.path.join(self.config.checkpoint_dir, "strategy_pool.pkl")
            if os.path.exists(pool_path):
                with open(pool_path, "rb") as f:
                    pool_data = pickle.load(f)
                self.strategy_pool.population = pool_data["population"]
                self.strategy_pool.fitness = pool_data["fitness"]
                self.strategy_pool.epoch = pool_data["epoch"]
                self.logger.info("Loaded strategy pool")
                
            # Load environment state
            env_path = os.path.join(self.config.checkpoint_dir, "env_state.pkl")
            if os.path.exists(env_path):
                with open(env_path, "rb") as f:
                    env_data = pickle.load(f)
                    
                # Restore environment state
                self.set_state(env_data["state"])
                
                # Restore module states
                for name, state in env_data["module_states"].items():
                    module = self.pipeline.get_module(name)
                    if module and hasattr(module, 'set_state'):
                        module.set_state(state)
                        
                self.logger.info("Loaded environment state")
                
        except Exception as e:
            self.logger.warning(f"Checkpoint load failed: {e}, starting fresh")
            
    # ═══════════════════════════════════════════════════════════════════
    #  Compatibility Attributes
    # ═══════════════════════════════════════════════════════════════════
    
    @property
    def balance(self):
        """Compatibility property for balance"""
        return self.market_state.balance
        
    @balance.setter
    def balance(self, value):
        self.market_state.balance = value
        
    @property
    def current_step(self):
        """Compatibility property for current step"""
        return self.market_state.current_step
        
    @current_step.setter
    def current_step(self, value):
        self.market_state.current_step = value
        
    @property
    def current_drawdown(self):
        """Compatibility property for drawdown"""
        return self.market_state.current_drawdown
        
    @current_drawdown.setter
    def current_drawdown(self, value):
        self.market_state.current_drawdown = value
        
    @property
    def open_positions(self):
        """Compatibility property for open positions"""
        return self.position_manager.open_positions
        
    @property
    def peak_balance(self):
        """Compatibility property for peak balance"""
        return self.market_state.peak_balance
        
    @peak_balance.setter
    def peak_balance(self, value):
        self.market_state.peak_balance = value
        
    @property
    def ep_step(self):
        """Compatibility property for episode step"""
        return self.market_state.current_step
        
    @ep_step.setter
    def ep_step(self, value):
        self.market_state.current_step = value
        
    @property
    def live_mode(self):
        """Compatibility property for live mode"""
        return self.config.live_mode
        
    @live_mode.setter
    def live_mode(self, value):
        self.config.live_mode = value
        
    @property
    def point_value(self):
        """Point value for each instrument"""
        if not hasattr(self, '_point_value'):
            self._point_value = {inst: 1.0 for inst in self.instruments}
        return self._point_value


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Module Export                                 ║
# ╚═════════════════════════════════════════════════════════════════════╝


        
    def _check_termination(self) -> bool:
        """Check if episode should terminate"""
        # Check balance
        if self.market_state.balance <= 0:
            self.logger.info("Episode terminated: balance depleted")
            return True
            
        # Check max steps
        if self.market_state.current_step >= self.config.max_steps:
            self.logger.info("Episode terminated: max steps reached")
            return True
            
        # Check data availability
        df = self.data[self.instruments[0]]["D1"]
        if self.market_state.current_step >= len(df) - 1:
            self.logger.info("Episode terminated: no more data")
            return True
            
        return False
        
    def _create_step_info(
        self,
        trades: List[Dict[str, Any]],
        reward: float,
        consensus: float
    ) -> Dict[str, Any]:
        """Create info dictionary for step"""
        return {
            "balance": round(self.market_state.balance, 2),
            "pnl": round(sum(t["pnl"] for t in trades), 2),
            "drawdown": round(self.market_state.current_drawdown, 4),
            "trades_executed": len(trades),
            "reward": round(reward, 4),
            "consensus": round(consensus, 4),
            "mode": self.mode_manager.get_stats(),
            "votes": self.episode_metrics.votes_log[-1] if self.episode_metrics.votes_log else {},
            "explanation": self.explainer.last_explanation if hasattr(self.explainer, 'last_explanation') else "",
            "memory": {
                "intuition_norm": float(np.linalg.norm(self.memory_compressor.intuition_vector)),
                "playbook_size": len(self.playbook_memory._features),
            },
            "step": self.market_state.current_step,
        }
        
    def _update_mode_manager(
        self,
        trades: List[Dict[str, Any]],
        pnl: float,
        consensus: float
    ):
        """Update trading mode manager"""
        # Determine trade result
        if trades:
            trade_result = "win" if pnl > 0 else "loss"
        else:
            trade_result = "hold"
            
        # Get current volatility
        vol = self._get_current_volatility()
        
        # Update mode
        self.mode_manager.step(
            trade_result=trade_result,
            pnl=float(pnl),
            consensus=float(consensus),
            volatility=float(vol),
            drawdown=self.market_state.current_drawdown,
        )
        
    def _handle_episode_end(self, last_pnl: float):
        """Handle episode termination"""
        # Calculate episode statistics
        if self.episode_metrics.pnls:
            # Create balance array
            balances = np.array(
                [self.config.initial_balance] +
                list(np.cumsum(self.episode_metrics.pnls) + self.config.initial_balance),
                dtype=np.float32
            )
            
            # Calculate returns
            returns = np.diff(balances) / (balances[:-1] + 1e-8)
            returns = np.nan_to_num(returns)
            
            # Calculate metrics
            mean_return = returns.mean()
            std_return = returns.std() + 1e-8
            sharpe = np.clip(mean_return / std_return * np.sqrt(250), -2.0, 5.0)
            
            # Calculate Sortino
            negative_returns = returns[returns < 0]
            downside_std = negative_returns.std() + 1e-8 if negative_returns.size > 0 else std_return
            sortino = np.clip(mean_return / downside_std * np.sqrt(250), -1.0, 3.0)
            
            # Log episode summary
            self.logger.info(
                f"Episode {self.episode_count} complete: "
                f"PnL={last_pnl:.2f}, Sharpe={sharpe:.3f}, "
                f"Sortino={sortino:.3f}, DD={self.market_state.current_drawdown:.2%}"
            )
            
            # Evaluate and evolve strategy pool
            self._evaluate_strategy_fitness(last_pnl, sharpe, sortino)
            
        # Save checkpoints
        try:
            self._save_checkpoints()
        except Exception as e:
            self.logger.error(f"Failed to save checkpoints: {e}")
            
    def _evaluate_strategy_fitness(self, pnl: float, sharpe: float, sortino: float):
        """Evaluate fitness of current strategy genome"""
        def fitness_function(genome: np.ndarray) -> float:
            sl_base, tp_base, vol_scale, regime_adapt = genome
            
            # Base fitness from performance metrics
            base_fitness = 0.4 * pnl + 0.3 * sharpe + 0.3 * sortino
            
            # Penalty for extreme drawdown
            dd_penalty = 0.5 * self.market_state.current_drawdown
            
            # Bonus for balanced risk parameters
            risk_bonus = 0.05 * (sl_base + tp_base) / 2.0
            
            # Bonus for adaptation capabilities
            adapt_bonus = 0.1 * (vol_scale + regime_adapt) / 2.0
            
            return float(base_fitness - dd_penalty + risk_bonus + adapt_bonus)
            
        # Evaluate population
        self.strategy_pool.evaluate_population(fitness_function)
        
        # Evolve strategies
        self.strategy_pool.evolve_strategies()
        
    def _create_reset_info(self) -> Dict[str, Any]:
        """Create info dictionary for reset"""
        return {
            "episode": self.episode_count,
            "starting_balance": self.market_state.balance,
            "starting_step": self.market_state.current_step,
            "max_steps": self.config.max_steps,
            "instruments": self.instruments,
            "genome": self.current_genome.tolist() if hasattr(self, 'current_genome') else [],
        }
        
    def _get_full_observation(self, data: Dict[str, Any]) -> np.ndarray:
        """Construct full observation vector from all modules"""
        # Get base observations from pipeline
        base = self.pipeline.step(data)
        
        # Validate base observation
        assert not np.any(np.isnan(base)), "NaN in base observation"
        
        # Get additional components
        components = [
            base,
            self.strategy_pool.get_observation_components(),
            self.meta_agent.get_observation_components(),
            self.meta_planner.get_observation_components(),
        ]
        
        # Add meta-RL components if available (it’s only created later)
        if hasattr(self, 'meta_rl') and hasattr(self.meta_rl, 'get_observation_components'):
            components.append(self.meta_rl.get_observation_components())
        else:
            components.append(np.zeros(4, dtype=np.float32))

            
        # Add other module components
        components.extend([
            self.long_term_memory.get_observation_components(),
            self.opp_enhancer.get_observation_components(),
            self.curriculum_planner.get_observation_components(),
            self.playbook_clusterer.get_observation_components(),
        ])
        
        # Concatenate all components
        obs = np.concatenate(components)
        
        # Ensure correct size
        if hasattr(self, 'observation_space'):
            expected_size = self.observation_space.shape[0]
            if obs.shape[0] < expected_size:
                obs = np.pad(obs, (0, expected_size - obs.shape[0]), constant_values=0)
            elif obs.shape[0] > expected_size:
                obs = obs[:expected_size]
                
        # Final validation
        assert not np.any(np.isnan(obs)), "NaN in final observation"
        
        return obs
        
    # ═══════════════════════════════════════════════════════════════════
    #  Public API Methods
    # ═══════════════════════════════════════════════════════════════════
    
    def get_instrument_correlations(self) -> Dict[Tuple[str, str], float]:
        """Calculate pairwise correlations between instruments"""
        correlations = {}
        
        for inst1, inst2 in combinations(self.instruments, 2):
            try:
                # Get returns for both instruments
                df1 = self.data[inst1]["D1"]["close"].pct_change().dropna()
                df2 = self.data[inst2]["D1"]["close"].pct_change().dropna()
                
                # Use last 100 bars
                returns1 = df1.iloc[-100:].values
                returns2 = df2.iloc[-100:].values
                
                # Calculate correlation
                if returns1.size > 0 and returns2.size > 0:
                    corr = np.corrcoef(returns1, returns2)[0, 1]
                    if np.isfinite(corr):
                        correlations[(inst1, inst2)] = float(corr)
                    else:
                        correlations[(inst1, inst2)] = 0.0
                else:
                    correlations[(inst1, inst2)] = 0.0
                    
            except Exception as e:
                self.logger.warning(f"Correlation calc failed for {inst1}/{inst2}: {e}")
                correlations[(inst1, inst2)] = 0.0
                
        return correlations
        
    def get_volatility_profile(self) -> Dict[str, float]:
        """Get current volatility for all instruments"""
        profile = {}
        
        for inst in self.instruments:
            try:
                df = self.data[inst]["D1"]
                if self.market_state.current_step < len(df):
                    vol = df.iloc[self.market_state.current_step]["volatility"]
                    profile[inst] = float(vol)
                else:
                    profile[inst] = 0.01
            except Exception:
                profile[inst] = 0.01
                
        return profile
        
    def get_genome_metrics(self) -> Dict[str, float]:
        """Get metrics for best strategy genome"""
        best_idx = int(np.argmax(self.strategy_pool.fitness))
        genome = self.strategy_pool.population[best_idx]
        sl_base, tp_base, vol_scale, regime_adapt = genome
        
        return {
            "sl_base": float(sl_base),
            "tp_base": float(tp_base),
            "vol_scale": float(vol_scale),
            "regime_adapt": float(regime_adapt),
            "fitness": float(self.strategy_pool.fitness[best_idx]),
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        pnls = self.episode_metrics.pnls
        
        if pnls:
            win_rate = float((np.array(pnls) > 0).mean())
            avg_pnl = float(np.mean(pnls))
            last_pnl = float(pnls[-1])
        else:
            win_rate = avg_pnl = last_pnl = 0.0
            
        return {
            "balance": self.market_state.balance,
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "last_pnl": last_pnl,
            "drawdown": self.market_state.current_drawdown,
            "trades_count": len(self.episode_metrics.trades),
        }
        
    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all modules"""
        status = {}

        all_modules = self.pipeline.modules + self.arbiter.members
        for module in all_modules:
            name = module.__class__.__name__

            # Get confidence if available
            confidence = 0.0
            if hasattr(module, 'confidence'):
                try:
                    # call module.confidence() without undefined args
                    confidence = float(module.confidence())
                except Exception:
                    confidence = 0.0

            status[name] = {
                "enabled": self.module_enabled.get(name, True),
                "confidence": confidence,
            }

        return status

        

        
    def _round_lot_size(self, size: float, symbol_info) -> float:
        """Round lot size to broker requirements"""
        step = symbol_info.volume_step or 0.01
        min_vol = symbol_info.volume_min or step
        max_vol = symbol_info.volume_max or 100.0
        
        # Round to step
        size = math.floor(size / step) * step
        
        # Apply limits
        size = max(min_vol, min(size, max_vol))
        
        return size
        
    def _get_regime_info(self) -> Dict[str, Any]:
        """Get current market regime information"""
        # Update fractal confirmation
        self.fractal_confirm.step(
            data_dict=self.data,
            current_step=self.market_state.current_step,
            theme_detector=self.theme_detector,
        )

        # Pull live balance if in live mode
        account_info = mt5.account_info()
        if account_info and hasattr(account_info, "balance"):
            self.market_state.balance = float(account_info.balance)
            self.market_state.peak_balance = max(
                self.market_state.peak_balance,
                self.market_state.balance
            )

        # Return the fully closed dict
        return {
            "label": self.fractal_confirm.label,
            "strength": self.fractal_confirm.regime_strength,
            "volatility": self.get_volatility_profile(),
            "correlations": self.get_instrument_correlations(),
        }

        
    def _finalize_step(
        self,
        trades: List[Dict[str, Any]],
        actions: np.ndarray,
        consensus: float
    ) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Finalize step with bookkeeping and reward calculation"""
        # Update episode metrics
        self.episode_metrics.trades.extend(trades)
        self.episode_metrics.pnls.extend(t["pnl"] for t in trades)
        self.episode_metrics.durations.extend(t.get("duration", 1) for t in trades)
        self.episode_metrics.drawdowns.append(self.market_state.current_drawdown)
        
        # Update memory systems
        self._update_memory_systems(trades)
        
        # Calculate step PnL
        step_pnl = sum(t["pnl"] for t in trades)
        
        # Update risk system
        self.risk_system.step(pnl=step_pnl)
        
        # Calculate reward
        reward = self._calculate_reward(trades, actions, step_pnl)
        
        # Apply no-trade penalty if needed
        if not trades:
            penalty = self.config.no_trade_penalty * (1 + self.market_state.current_drawdown)
            reward -= penalty
            self.logger.debug(f"Applied no-trade penalty: {penalty:.4f}")
            
        # Update arbiter with reward feedback
        self.arbiter.update_reward(reward)
        
        # Update meta-learning systems
        self._update_meta_systems(step_pnl, trades)
        
        # Run monitoring systems
        self._run_monitoring_systems(trades, step_pnl)
        
        # Get next observation
        obs = self._get_next_observation(actions, trades, step_pnl)
        
        # Check termination conditions
        terminated = self._check_termination()
        
        # Create info dict
        info = self._create_step_info(trades, reward, consensus)
        
        # Update mode manager
        self._update_mode_manager(trades, step_pnl, consensus)
        
        # Increment step counter
        self.market_state.current_step += 1
        
        # Handle episode end
        if terminated:
            self._handle_episode_end(step_pnl)
            
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
                
            if all(len(p) > self.risk_system.var_window for p in price_dict.values()):
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
            start = max(0, self.market_state.current_step - 7)
            end = self.market_state.current_step
            return df["close"].iloc[start:end].values.astype(np.float32)
        except Exception:
            return np.zeros(7, dtype=np.float32)
            
    def _calculate_instrument_confidence(self, instrument: str) -> float:
        """Calculate confidence score for an instrument"""
        if not hasattr(self, '_last_votes') or not self._last_votes:
            return 0.0
            
        confidences = []
        for tf in ["H1", "H4", "D1"]:
            key = (instrument, tf)
            if key in self._last_votes:
                conf_values = list(self._last_votes[key].values())
                if conf_values:
                    confidences.extend(conf_values)
                    
        return float(np.mean(confidences)) if confidences else 0.0
        
    def _calculate_position_size(
        self,
        instrument: str,
        intensity: float,
        volatility: float
    ) -> float:
        """Calculate position size with all safety checks"""
        # 1) Get raw size from position manager (now the call is properly closed)
        raw_size = self.position_manager.calculate_size(
            volatility=volatility,
            intensity=intensity,  # Check confidence threshold
            confidence=self._calculate_instrument_confidence(instrument)
        )

        # 2) Enforce minimum per-instrument confidence
        confidence = self._calculate_instrument_confidence(instrument)
        if confidence < self.config.min_inst_confidence:
            self.logger.debug(
                f"Skipping {instrument}: confidence {confidence:.2f} "
                f"< {self.config.min_inst_confidence}"
            )
            return 0.0

        # 3) Sanity-check the raw size
        if not np.isfinite(raw_size) or raw_size <= 0.0:
            self.logger.debug(f"Invalid raw_size {raw_size} for {instrument}, setting to 0")
            return 0.0

        # 4) Cap by maximum dollar exposure
        df = self.data[instrument]["D1"]
        price = float(df.iloc[self.market_state.current_step]["close"])
        point_value = self.point_value.get(instrument, 1.0)

        max_dollars = self.market_state.balance * self.position_manager.max_pct
        max_lots = max_dollars / (price * point_value + 1e-12)
        if raw_size > max_lots:
            self.logger.info(
                f"{instrument}: capping size from {raw_size:.3f} to {max_lots:.3f} lots"
            )
            raw_size = max_lots

        return raw_size


        
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
        """Execute a live trade via MetaTrader5"""
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
            
        # Register position
        self.position_manager.open_positions[instrument] = {
            "ticket": result.order,
            "side": 1 if intensity > 0 else -1,
            "lots": size,
            "price_open": price,
            "peak_profit": 0.0,
        }
        