# envs/ppo_env.py
from __future__ import annotations

# ─────────────────────────── Std-lib ──────────────────────────────────
from itertools import combinations
import math
import json, os, copy, random, logging, pickle
from typing import Any, Dict, List

# ───────────────────────── Third-party ────────────────────────────────
import numpy as np
import pandas as pd
import torch
import gymnasium as gym
from gymnasium import spaces

# ──────────────────────── Internal modules ────────────────────────────
from modules.core.core import Module
from modules.features.feature import AdvancedFeatureEngine, MultiScaleFeatureEngine
from modules.position.position import PositionManager
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

# NEW — adaptive mode, risk monitoring, sentiment
from modules.trading_modes.trading_mode import TradingModeManager
from modules.risk.risk_monitor import (
    ActiveTradeMonitor, CorrelatedRiskController, DrawdownRescue,
    ExecutionQualityMonitor, AnomalyDetector,
)
from modules.external.news_sentiment import NewsSentimentModule

# ─────────────────────────── Logger setup for "SGP" ──────────────────────────
# Capture all "SGP" logs into a separate file, and prevent console output.
sgp_logger = logging.getLogger("SGP")
if not sgp_logger.handlers:
    sgp_handler = logging.FileHandler("logs/sgp.log")
    sgp_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    sgp_handler.setFormatter(sgp_formatter)
    sgp_logger.addHandler(sgp_handler)
sgp_logger.propagate = False
sgp_logger.setLevel(logging.INFO)

# ╔═════════════════════════════════════════════════════════════════════╗
# ║                         Helper utilities                           ║
# ╚═════════════════════════════════════════════════════════════════════╝
def _ensure_dir(path: str):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                     Processing pipeline wrapper                    ║
# ╚═════════════════════════════════════════════════════════════════════╝
class TradingPipeline:
    def __init__(self, modules: List[Module]):
        self.modules: List[Module] = modules

    def reset(self):
        for m in self.modules:
            m.reset()

    def step(self, data: Dict[str, Any]) -> np.ndarray:
        """
        Call every module in sequence, respecting env.module_enabled,
        and concatenate their observation components.
        """
        env = data.get("env")
        parts: List[np.ndarray] = []
        for m in self.modules:
            if env and not env.module_enabled.get(m.__class__.__name__, True):
                continue
            sig = m.step.__code__.co_varnames[1 : m.step.__code__.co_argcount]
            m.step(**{k: data[k] for k in sig if k in data})
            parts.append(m.get_observation_components())
        obs = np.concatenate(parts) if parts else np.zeros(0, np.float32)
        return (obs - obs.mean()) / (obs.std() + 1e-8)


# ╔═════════════════════════════════════════════════════════════════════╗
# ║                        Core Gym-style Env                          ║
# ╚═════════════════════════════════════════════════════════════════════╝
class EnhancedTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    # ──────────────────────────── INIT ────────────────────────────────
    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        initial_balance: float = 3000.0,
        max_steps: int = 200,
        debug: bool = False,
        no_trade_penalty: float = 0.3,
        init_seed: int = 0,
        checkpoint_dir: str = "checkpoints",
    ):
        super().__init__()

        # ──────────────── Logger setup for EnhancedTradingEnv ─────────────────
        self.logger = logging.getLogger("EnhancedTradingEnv")
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/env.log")
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # Prevent propagation to the root logger (so logs won’t also print to console)
        self.logger.propagate = False
        self.logger.info("Logger initialized")

        # Episode parameters
        self.initial_balance = float(initial_balance)
        self.max_steps = int(max_steps)
        self.max_holding_period = self.max_steps  # upper bound for simulated hold

        # Deterministic seeding
        random.seed(init_seed)
        np.random.seed(init_seed)
        torch.manual_seed(init_seed)
        torch.cuda.manual_seed_all(init_seed)

        self.debug = debug
        self.no_trade_penalty = no_trade_penalty
        self.episode_count = 0
        self.consensus_min = 0.15    # Reduced from 0.2
        self.consensus_max = 0.5     # Reduced from 0.6
        self.max_episodes   = 10000
        self.ep_step = 0
        # Data
        self.orig_data = data_dict
        self.data = copy.deepcopy(data_dict)
        self.instruments = list(data_dict.keys())

        # ────────────────── Module instantiation ──────────────────────
        base_afe = AdvancedFeatureEngine(debug=debug)
        self.feature_engine    = MultiScaleFeatureEngine(base_afe, 32, debug)
        self.position_manager = PositionManager(
            initial_balance=self.initial_balance,
            instruments=self.instruments,
            debug=debug
        )
        self.position_manager.set_env(self)  # <--- CRITICAL LINE

        self.open_positions    = getattr(self.position_manager, "positions", [])
        self.reward_shaper     = RiskAdjustedReward(self.initial_balance, env=self, debug=debug)
        action_dim = 2 * len(self.instruments)
        self.liquidity_layer = LiquidityHeatmapLayer(
            action_dim=action_dim,
            debug=debug
        )
        self.risk_controller   = DynamicRiskController({
            "freeze_counter": 0, "freeze_duration": 5,
            "vol_history_len": 100, "dd_threshold": 0.2,
            "vol_ratio_threshold": 1.5,
        }, debug=debug, action_dim=action_dim)

        self.meta_rl = None

        # Memory / analysis
        self.mistake_memory    = MistakeMemory(interval=10, n_clusters=3, debug=debug)
        self.memory_compressor = MemoryCompressor(50, 5, debug=debug)
        self.replay_analyzer   = HistoricalReplayAnalyzer(10, debug=debug)
        self.shadow_sim        = ShadowSimulator(debug=debug)
        self.playbook_memory   = PlaybookMemory(debug=debug)
        self.strategy_intros   = StrategyIntrospector(debug=debug)
        self.curriculum_planner= CurriculumPlannerPlus(debug=debug)
        self.memory_budget     = MemoryBudgetOptimizer(1000, 500, 300, debug=debug)
        self.role_coach        = RoleCoach(debug=debug)
        self.opponent_sim      = OpponentSimulator(debug=debug)

        # Market context
        self.theme_detector    = MarketThemeDetector(self.instruments, 4, 100, debug=debug)
        self.fractal_confirm   = FractalRegimeConfirmation(100, debug=debug)
        self.time_risk_scaler  = TimeAwareRiskScaling(debug=debug)

        self.visualizer        = VisualizationInterface(debug=debug)

        # Long-horizon & meta
        self.playbook_clusterer= PlaybookClusterer(5, debug=debug)
        self.long_term_memory  = NeuralMemoryArchitect(32, 4, 500, debug=debug)
        self.meta_agent        = MetaAgent(debug=debug)
        self.meta_planner      = MetaCognitivePlanner(debug=debug)
        self.bias_auditor      = BiasAuditor(debug=debug)
        self.opp_enhancer      = OpponentModeEnhancer(debug=debug)
        self.thesis_engine     = ThesisEvolutionEngine(debug=debug)
        self.regime_matrix     = RegimePerformanceMatrix(debug=debug)
        self.trade_map_vis     = TradeMapVisualizer(debug=debug)
        self.trade_thesis      = TradeThesisTracker(debug=debug)
        self.world_model       = RNNWorldModel(2, debug=debug)
        self.explainer         = ExplanationGenerator(debug=debug)

        # Compliance
        self.compliance        = ComplianceModule()

        # Portfolio risk
        self.risk_system       = PortfolioRiskSystem(50, 0.2, debug=debug)

        # NEW — adaptive mode, risk monitoring, sentiment
        self.mode_manager      = TradingModeManager(initial_mode="safe", window=50)
        self.active_monitor    = ActiveTradeMonitor(max_duration=self.max_holding_period)
        self.corr_controller   = CorrelatedRiskController(max_corr=0.8)
        self.dd_rescue         = DrawdownRescue(dd_limit=0.3)
        self.exec_monitor      = ExecutionQualityMonitor()
        self.anomaly_detector  = AnomalyDetector()
        self.news_sentiment    = NewsSentimentModule(enabled=False, debug=debug)

        # Evolution pool
        self.strategy_pool     = StrategyGenomePool(20, debug=debug)

        # ────────────── Analysis pipeline (observation) ───────────────
        core_modules = [
            self.feature_engine, self.compliance, self.risk_system,
            self.theme_detector, self.time_risk_scaler, self.liquidity_layer,
            self.strategy_intros, self.curriculum_planner, self.memory_budget,
            self.bias_auditor, self.opp_enhancer, self.thesis_engine,
            self.regime_matrix, self.trade_thesis,
            self.mode_manager, self.active_monitor, self.corr_controller,
            self.dd_rescue, self.exec_monitor, self.anomaly_detector,
            self.news_sentiment,
        ]
        self.pipeline = TradingPipeline(core_modules)

        # Toggle dictionary
        self.module_enabled = {
            m.__class__.__name__: True
            for m in core_modules + [
                self.position_manager, self.risk_controller,
                self.mistake_memory, self.memory_compressor,
            ]
        }

        # Initial obs (needed before defining spaces)
        obs, _ = self.reset(seed=init_seed)

        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (2 * len(self.instruments),), np.float32)
        self.action_space.seed(init_seed)
        self.action_dim = self.action_space.shape[0]

        self.meta_rl = MetaRLController(obs.size, self.action_space.shape[0])
        self.meta_rl.obs_dim = obs.size

        # Voting committee (unchanged)
        committee = [self.position_manager, self.risk_controller, self.liquidity_layer]
        init_w = np.full(len(committee), 1 / len(committee), np.float32)
        self.consensus  = ConsensusDetector(len(committee), 0.7)
        self.collusion  = CollusionAuditor(len(committee), 0.95)
        self.haligner   = TimeHorizonAligner(
            [getattr(m, "decision_horizon", 0) for m in committee]
        )
        self.arbiter    = StrategyArbiter(
            committee, init_w,
            action_dim=self.action_space.shape[0],
            adapt_rate=0.01,
            consensus=self.consensus,
            collusion=self.collusion,
            horizon_aligner=self.haligner,
        )
        self.alt_sampler = AlternativeRealitySampler(len(committee))

        # Live-trading bookkeeping
        self.live_mode = False
        info = None  # Placeholder for live mode
        real_bal = float(info.balance) if info and hasattr(info, "balance") else self.initial_balance
        self.balance = self.peak_balance = real_bal

        self.current_step    = 0
        self.current_drawdown= 0.0
        self.sl_multiplier   = self.tp_multiplier = 1.0

        # Episode-level logs
        self.votes_log: List[Dict[str, Any]] = []
        self.reasoning_trace: List[str] = []
        self._ep_pnls: List[float] = []
        self._ep_durations: List[int] = []
        self._ep_drawdowns: List[float] = []

        # Checkpointing
        self.ckpt_dir = checkpoint_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self._maybe_load_checkpoints()

        # Point value for simulated PnL
        self.point_value = {instr: 1.0 for instr in self.instruments}

    # ==================================================================
    #  RESET
    # ==================================================================
    def sanitize_obs(self, obs: np.ndarray) -> np.ndarray:
        return np.nan_to_num(obs, nan=0.0)  # Replace NaN with 0 or another default value

    def _blend_committee_votes(
        self,
        blended_by_sym_tf: dict[tuple[str, str], np.ndarray],
    ) -> np.ndarray:
        """
        Average the per-timeframe blends produced by the Arbiter into one
        (size, duration) pair per instrument.

        Returns
        -------
        np.ndarray, shape (action_dim,)
        """
        big_action = np.zeros(self.action_space.shape[0], np.float32)

        for i, inst in enumerate(self.instruments):
            per_tf = []
            for tf in ("H1", "H4", "D1"):
                key = (inst, tf)
                if key not in blended_by_sym_tf:
                    continue
                per_tf.append(blended_by_sym_tf[key][2*i:2*i+2])

            if per_tf:   # mean across whichever TFs we actually have
                big_action[2*i:2*i+2] = np.mean(per_tf, axis=0)

        # final sanity-check
        big_action = np.nan_to_num(big_action, copy=False)

        return big_action

    def reset(self, *, seed: int | None = None, options=None):
        if not hasattr(self, "_ep_pnls"):
            self._ep_pnls, self._ep_durations, self._ep_drawdowns = [], [], []

        # Deterministic seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if hasattr(self, "action_space"):
                self.action_space.seed(seed)

        # Restore data & balances
        self.episode_count += 1
        self.data = copy.deepcopy(self.orig_data)
        info = None  # Placeholder for live mode
        real_balance = float(info.balance) if info and hasattr(info, "balance") else self.initial_balance
        self.balance = self.peak_balance = real_balance

        # Index bookkeeping
        self.current_step = random.randint(100, len(self.data["XAU/USD"]["D1"]) - self.max_steps - 1)
        self.ep_step = 0
        self.current_drawdown = 0.0
        self.sl_multiplier = self.tp_multiplier = 1.0

        # Reset all modules and meta/neuro modules
        core_modules = [
            self.feature_engine, self.position_manager, self.reward_shaper,
            self.risk_controller, self.mistake_memory, self.memory_compressor,
            self.replay_analyzer, self.shadow_sim, self.playbook_memory,
            self.strategy_intros, self.curriculum_planner, self.memory_budget,
            self.role_coach, self.opponent_sim, self.theme_detector,
            self.visualizer, self.fractal_confirm, self.time_risk_scaler,
            self.liquidity_layer, self.playbook_clusterer, self.long_term_memory,
            self.meta_agent, self.meta_planner, self.bias_auditor,
            self.opp_enhancer, self.thesis_engine, self.regime_matrix,
            self.trade_map_vis, self.trade_thesis, self.world_model,
            self.compliance, self.risk_system, self.explainer,
            self.mode_manager, self.active_monitor, self.corr_controller,
            self.dd_rescue, self.exec_monitor, self.anomaly_detector,
            self.news_sentiment,
        ]
        for attr in ("meta_rl", "arbiter", "consensus", "collusion", "haligner"):
            mod = getattr(self, attr, None)
            if mod is not None:
                core_modules.append(mod)
        for m in core_modules:
            if hasattr(m, "reset"):
                m.reset()

        # ----------- NEURO/EVOLUTIONARY RESET -----------
        # Select a new genome for this episode (neuro-evolution)
        self.strategy_pool.select_genome("random")
        self.current_genome = self.strategy_pool.active_genome.copy()
        # Reset meta/agent/planner modules
        self.meta_agent.reset()
        self.meta_planner.reset()
        if getattr(self, "meta_rl", None) is not None:
             self.meta_rl.reset()

        self.memory_vector = self.long_term_memory.retrieve(None)
        self.opp_enhancer.reset()
        self.curriculum_planner.reset()
        self.playbook_clusterer.reset()

        # Pipeline reset
        self.position_manager.max_pct = self.position_manager.default_max_pct
        self.pipeline.reset()

        # Adaptive episode length
        vol = self.data["XAU/USD"]["D1"].iloc[self.current_step]["volatility"]
        self.max_steps = int(np.clip(200 * (0.5 / max(vol, 1e-8)), 50, 400))

        # First observation includes all meta/neuro features
        obs = self._get_full_observation(self._dummy_input())
        obs = self.sanitize_obs(obs)
        if getattr(self, "meta_rl", None) is not None:
            self.meta_rl.last_embedding = np.zeros_like(obs)

        # House-keeping
        self.votes_log = [{}]
        self.reasoning_trace = [""]
        self._ep_pnls.clear()
        self._ep_durations.clear()
        self._ep_drawdowns.clear()

        self.logger.debug(f"Environment reset: initial balance={self.balance}, max steps={self.max_steps}")

        return obs, {}

    # ------------------------------------------------------------------
    #  Dummy input for very first observation
    # ------------------------------------------------------------------
    def _dummy_input(self) -> Dict[str, Any]:
        return {
            "env":         self,
            "price_h1":    np.zeros(7, np.float32),
            "price_h4":    np.zeros(7, np.float32),
            "price_d1":    np.zeros(7, np.float32),
            "actions":     np.zeros(2 * len(self.instruments), np.float32),
            "trades":      [],
            "open_trades": [],
            "drawdown":    0.0,
            "memory":      np.zeros(32, np.float32),
            "pnl":         0.0,
            "correlations": {},
        }

    # ==================================================================
    #  STEP
    # ==================================================================
    def step(self, actions: np.ndarray):
        self.logger.debug(f"Actions before adjustment: {actions}")
        self.logger.debug(f"Agent step: current_step={self.current_step}, balance={self.balance}")
        self.trades = []  # Reset trades for this step

        try:
            # ── MetaRL adjustment ───────────────────────────────────────
            obs_for_meta = self._get_full_observation(self._dummy_input())
            obs_tensor = torch.tensor(obs_for_meta, dtype=torch.float32, device="cpu").unsqueeze(0)
            meta_action_out = self.meta_rl.act(obs_tensor)

            # Handle dict vs tensor output
            if isinstance(meta_action_out, dict):
                meta_action = meta_action_out["action"]
            else:
                meta_action = meta_action_out

            if isinstance(meta_action, torch.Tensor):
                meta_action = meta_action.detach().cpu().numpy()

            meta_action = np.asarray(meta_action).reshape(-1)
            if meta_action.shape[0] != self.action_dim:
                raise ValueError(f"meta_action shape mismatch: {meta_action.shape} (expected {self.action_dim})")

            # Apply meta_action and mode scaling
            actions = actions + meta_action * self.current_genome[2]
            actions = np.clip(actions, self.action_space.low, self.action_space.high)
            mode_coef = {"safe": 0.5, "normal": 1.0, "aggressive": 1.25, "extreme": 1.5}
            cur_mode = self.mode_manager.get_mode()
            actions = actions * mode_coef.get(cur_mode, 1.0)

            # ── Risk / correlation checks ─────────────────────────────────
            self.active_monitor.step(open_trades=self.open_positions)
            corr_dict = self.get_instrument_correlations()
            high_corr = self.corr_controller.step(correlations=corr_dict)
            if high_corr:
                self.position_manager.max_pct *= 0.5
                self.logger.info(f"High correlation detected → halving max_pct to {self.position_manager.max_pct:.4f}")

            if self.dd_rescue.step(current_drawdown=self.current_drawdown):
                self.logger.info("Drawdown rescue → skipping trades")
                return self._finalize_step([], actions, corr_dict, vol0=None, reward=0)

            # ── Adjust risk controller ────────────────────────────────────
            df0 = self.data[self.instruments[0]]["D1"]
            vol0 = df0.iloc[self.current_step]["volatility"]
            self.risk_controller.adjust_risk({
                "drawdown":   self.current_drawdown,
                "volatility": float(vol0),
            })
            rcoef = float(self.risk_controller.get_observation_components()[0])
            base_pct = getattr(self.position_manager, "default_max_pct", None) or self.position_manager.max_pct
            cap = min(base_pct * rcoef, base_pct)
            self.position_manager.max_pct = cap
            self.logger.debug(f"Applied max_pct cap: {cap:.4f} (base={base_pct:.4f}, rcoef={rcoef:.4f}, high_corr={high_corr})")

            # ── Committee vote blending ───────────────────────────────────
            votes_by_sym_tf   = {}
            blended_by_sym_tf = {}
            committee = [m.__class__.__name__ for m in self.arbiter.members]

            for inst in self.instruments:
                for tf in ("H1", "H4", "D1"):
                    hist = (
                        self.data[inst][tf]["close"]
                        .iloc[self.current_step - 7 : self.current_step]
                        .values.astype(np.float32)
                    )
                    obs_tf = self._get_full_observation({
                        "env":       self,
                        f"price_{tf.lower()}": hist,
                        "actions":   actions,
                    })
                    blend = self.arbiter.propose(obs_tf)

                    alpha = (
                        self.arbiter.last_alpha.copy()
                        if self.arbiter.last_alpha is not None
                        else np.zeros(self.action_dim, dtype=np.float32)
                    )

                    votes_by_sym_tf[(inst, tf)]   = dict(zip(committee, alpha.tolist()))
                    blended_by_sym_tf[(inst, tf)] = blend

            actions = self._blend_committee_votes(blended_by_sym_tf)

            # ── Regime confirmation & explanation ─────────────────────────
            regime_label, _ = self.fractal_confirm.step(
                data_dict=self.data,
                current_step=self.current_step,
                theme_detector=self.theme_detector,
            )
            self.explainer.step(
                actions,
                np.mean([list(v.values()) for v in votes_by_sym_tf.values()], axis=0),
                committee,
                regime=regime_label,
                volatility=self.get_volatility_profile(),
                drawdown=self.current_drawdown,
                genome_metrics=self.get_genome_metrics(),
            )

            self.logger.info(f"Proposed trades after arbiter blending: {actions}")
            merged = {
                **{"big_" + m: w for m, w in zip(committee, self.arbiter.weights.tolist())},
                **{
                    f"{inst}_{tf}_{mod}": w
                    for (inst, tf), vv in votes_by_sym_tf.items()
                    for mod, w in vv.items()
                }
            }
            self.votes_log.append(merged)
            self.reasoning_trace.append(self.explainer.last_explanation)

            os.makedirs("logs", exist_ok=True)
            with open("logs/votes_history.json", "w") as fp:
                json.dump(self.votes_log, fp, indent=2)

            # ── Dynamic consensus threshold ───────────────────────────────
            frac = min(self.episode_count / self.max_episodes, 1.0)
            thr  = self.consensus_min + (self.consensus_max - self.consensus_min) * frac
            alpha = (
                self.arbiter.last_alpha
                if self.arbiter.last_alpha is not None
                else np.zeros(self.action_dim, np.float32)
            )
            cons = float(np.mean(alpha))
            if cons < thr:
                self.logger.info(f"[DYN] Consensus {cons:.2f} < {thr:.2f} → skipping all trades")
                return self._finalize_step([], actions, corr_dict, vol0, reward=0)

            # ── Execute trades for each instrument ─────────────────────────
            trades = []
            for i, inst in enumerate(self.instruments):
                intensity = actions[2*i]
                duration_norm = actions[2*i + 1]

                self.logger.debug(f"Calling _execute_trade for {inst} with intensity={intensity:.4f}, duration_norm={duration_norm:.4f}")
                tr = self._execute_trade(inst, intensity, duration_norm)
                self.logger.debug(f"_execute_trade returned: {tr!r}")

                if tr:
                    passed = self.compliance.validate_trade(tr, self)
                    self.logger.debug(f"Compliance check for {inst}: {passed}")
                    if passed:
                        tr["explanation"]     = self.explainer.last_explanation
                        tr["votes_by_sym_tf"] = votes_by_sym_tf
                        trades.append(tr)
                        self.logger.info(f"Executed trade for {inst}: {tr}")
                    else:
                        self.logger.debug(f"Dropped trade for {inst} due to compliance.")
                else:
                    self.logger.debug(f"No trade object returned for {inst} (empty dict)")

            # Assign trades before finalizing
            self.trades = trades
            self.logger.info(f"Final trades list for this step: {self.trades}")

            # ── Neuro/meta logging ───────────────────────────────────────
            pnl = sum(t["pnl"] for t in trades)
            self.meta_agent.record(pnl)
            self.meta_planner.record_episode({"pnl": pnl, "drawdown": self.current_drawdown})
            self.opp_enhancer.step(trades=trades, pnl=pnl)
            self.curriculum_planner.step(result={"pnl": pnl, "obs": obs_for_meta})
            self.playbook_clusterer.step(trades=trades, obs=obs_for_meta)

            # ── Finalize step (compute reward, next obs, etc.) ───────────
            obs, reward, terminated, truncated, info = (
                self._finalize_step(trades, actions, corr_dict, vol0, reward=0)
            )
            self.ep_step += 1

            terminated = (
                self.balance <= 0
                or self.ep_step >= self.max_steps
                or self.current_step >= len(df0) - 1
            )

            if obs.size == self.meta_rl.obs_dim:
                self.meta_rl.record_step(obs, reward)

            if terminated:
                self._finish_episode(pnl)

            return obs, reward, terminated, truncated, info

        except Exception:
            self.logger.exception("Error in step()")
            raise

    def _finalize_step(self, trades, actions, corr_dict, vol0, reward):
        # Log entry to _finalize_step, including current trades list
        self.logger.info(f"Entering _finalize_step: current reward={reward}, no_trade_penalty={self.no_trade_penalty}")
        self.logger.debug(f" → Trades passed into _finalize_step: {trades!r}")

        # Record PnL, durations, and drawdowns
        self._ep_pnls.extend(t["pnl"] for t in trades)
        self._ep_durations.extend(t.get("duration", 1) for t in trades)
        self._ep_drawdowns.append(self.current_drawdown)

        # Update mistake memory and compress (or store “empty” if no trades)
        self.mistake_memory.step(trades=trades)
        if trades:
            self.memory_compressor.compress(self.current_step, trades)
        else:
            last_emb = getattr(self.feature_engine, "last_embedding", None)
            emb = last_emb if last_emb is not None else np.zeros_like(self.memory_compressor.intuition_vector)
            self.memory_compressor.compress(self.current_step, [{"features": emb}])
        self.memory_compressor.intuition_vector = np.ones_like(self.memory_compressor.intuition_vector)
        self.playbook_clusterer._ready = True

        # Update balance based on this step’s PnL
        pnl = sum(t["pnl"] for t in trades)
        self.balance += pnl
        self.peak_balance = max(self.peak_balance, self.balance)
        self.current_drawdown = max(
            (self.peak_balance - self.balance) / (self.peak_balance + 1e-8), 0.0
        )
        self.risk_system.step(pnl=pnl)

        # Fractal confirmation may adjust TP/SL multipliers
        label, strength = self.fractal_confirm.step(
            data_dict=self.data, current_step=self.current_step, theme_detector=self.theme_detector
        )
        if label == "trending":
            self.tp_multiplier *= 1.2
            self.sl_multiplier *= 0.8
        elif label == "volatile":
            self.tp_multiplier *= 0.8
            self.sl_multiplier *= 1.2

        # Compute the reward via RiskAdjustedReward
        reg_onehot = np.full_like(self.reward_shaper.regime_weights, strength, np.float32)
        reward = self.reward_shaper.step(
            self.balance, trades, self.current_drawdown,
            regime_onehot=reg_onehot, actions=actions
        )

        # Possibly add a replay bonus
        reward += self.replay_analyzer.maybe_replay(self.current_step)

        # ── DEBUG: Check if we are about to apply no‐trade penalty ─────
        self.logger.debug(f" → After reward shaping, trades = {trades!r}")
        if not trades:
            self.logger.debug(" → No trades detected this step ⇒ applying no_trade_penalty")
            reward -= self.no_trade_penalty * (1 + self.current_drawdown)

        # Tell arbiter the final reward for this step
        self.arbiter.update_reward(reward)

        # Build next‐step observation
        df0 = self.data[self.instruments[0]]["D1"]
        hist = df0["close"].iloc[self.current_step - 7 : self.current_step].values.astype(np.float32)
        mem = self.long_term_memory.retrieve(self.feature_engine.last_embedding)
        obs = self._get_full_observation({
            "env": self,
            "price_h1": hist,
            "price_h4": hist,
            "price_d1": hist,
            "actions": actions,
            "trades": trades,
            "open_trades": self.open_positions,
            "drawdown": self.current_drawdown,
            "memory": mem,
            "pnl": pnl,
            "correlations": corr_dict
        })

        # Post‐processing monitors
        self.exec_monitor.step(trade_executions=trades)
        self.anomaly_detector.step(pnl=pnl, obs=obs)
        if obs.size == self.meta_rl.obs_dim:
            self.meta_rl.record_step(obs, reward)

        terminated = (
            self.balance <= 0
            or self.current_step >= self.max_steps
            or self.current_step >= len(df0) - 1
        )
        if terminated:
            self._finish_episode(pnl)

        info = {
            "balance": round(self.balance, 2),
            "pnl": round(pnl, 2),
            "drawdown": round(self.current_drawdown, 4),
            "terminated": terminated,
            "mode": self.mode_manager.get_stats(),
            "votes": self.votes_log[-1],
            "big_action": actions.tolist(),
            "reason": self.explainer.last_explanation,
            "memory": {
                "intuition_norm": float(np.linalg.norm(self.memory_compressor.intuition_vector)),
                "playbook_size": len(self.playbook_memory._features)
            }
        }

        v0 = vol0 if vol0 is not None else self.get_volatility_profile().get(self.instruments[0], 0.0)
        alpha = getattr(self.arbiter, "last_alpha", None)
        cons = float(np.mean(alpha)) if isinstance(alpha, (list, np.ndarray)) else 0.0
        self.mode_manager.step(
            trade_result="win" if pnl > 0 else "loss",
            pnl=float(pnl),
            consensus=cons,
            volatility=v0,
            drawdown=self.current_drawdown,
        )
        self.theme_detector.fit_if_needed(self.data, self.current_step)
        self.current_step += 1

        return obs, float(reward), terminated, False, info

    # ==================================================================
    #  Helper: full observation vector construction
    # ==================================================================
    def _get_full_observation(self, data: Dict[str, Any]) -> np.ndarray:
        base = self.pipeline.step(data)
        pool = self.strategy_pool.get_observation_components()
        meta_agent = self.meta_agent.get_observation_components()
        meta_planner = self.meta_planner.get_observation_components()
        meta_rl = self.meta_rl.get_observation_components() if hasattr(self.meta_rl, "get_observation_components") else np.zeros(4)
        memory_vec = self.long_term_memory.get_observation_components()
        opponent = self.opp_enhancer.get_observation_components()
        curriculum = self.curriculum_planner.get_observation_components()
        playbook = self.playbook_clusterer.get_observation_components()

        # Concatenate all meta/neuro vectors
        obs = np.concatenate([
            base, pool, meta_agent, meta_planner, meta_rl, memory_vec, opponent, curriculum, playbook
        ])
        expected_dim = getattr(self, "observation_space", None)
        if expected_dim is not None:
            expected_dim = expected_dim.shape[0]
            if obs.shape[0] < expected_dim:
                obs = np.pad(obs, (0, expected_dim - obs.shape[0]), constant_values=0)
            elif obs.shape[0] > expected_dim:
                obs = obs[:expected_dim]
        return obs

    # ==================================================================
    #  Trade execution (live / simulated)
    # ==================================================================
    def _execute_trade(
        self,
        instrument: str,
        intensity: float,
        duration_norm: float,
    ) -> dict:
        """
        Execute a single trade (live or simulated), with NaN/Inf protection
        and an enforced cap at max_pct of account equity.
        """
        df = self.data[instrument]["D1"]
        if self.current_step >= len(df):
            return {}
        bar = df.iloc[self.current_step]

        # ─── Price & volatility ───────────────────────────────────────:
        price = float(bar["close"])
        raw_vol = bar.get("volatility", 0.0)
        vol = float(np.nan_to_num(raw_vol, nan=self.position_manager.min_volatility))
        vol = max(vol, self.position_manager.min_volatility)

        # ─── Position sizing ──────────────────────────────────────────
        raw_size = self.position_manager.calculate_size(
            volatility=vol,
            intensity=float(np.nan_to_num(intensity, nan=0.0)),
            balance=float(np.nan_to_num(self.balance, nan=self.position_manager.initial_balance)),
            drawdown=float(np.nan_to_num(self.current_drawdown, nan=0.0)),
        )
        if not np.isfinite(raw_size):
            raw_size = 0.0
        sign = np.sign(raw_size)
        size = abs(raw_size)

        # ─── Enforce max_pct cap ──────────────────────────────────────
        point_val = self.point_value.get(instrument, 1.0)
        max_dollars = self.balance * self.position_manager.max_pct
        max_lots = max_dollars / (price * point_val + 1e-12)
        if size > max_lots:
            self.logger.info(
                f"[{instrument}] raw size {size:.3f} lots exceeds cap "
                f"{max_lots:.3f} lots → capping"
            )
            size = max_lots

        self.logger.info(
            f"[{instrument}] intensity={intensity:.3f} | "
            f"max_pct={self.position_manager.max_pct:.3f} | "
            f"raw_size={raw_size:.4f} lots | capped_size={size:.4f} lots"
        )
         
        # ─── Live vs Simulation ──────────────────────────────────────
        if self.live_mode:
            return self._execute_live_trade(instrument, size, intensity)
        
        # ── SIMULATION / BACKTEST BRANCH ──────────────────────────────
        self.logger.info(f"[SIM] TRADE START {instrument} "
                         f"{'BUY' if sign > 0 else 'SELL'} {size:.3f} @ {price:.4f}")
        hold_steps = max(int(duration_norm * self.max_holding_period), 1)
        exit_idx = min(self.current_step + hold_steps, len(df) - 1)
        exit_price = float(df.iloc[exit_idx]["close"])

        # ─── PnL calculation ──────────────────────────────────────────
        if sign > 0:
            pnl = (exit_price - price) * size * point_val
        else:
            pnl = (price - exit_price) * size * point_val
        pnl = float(np.nan_to_num(pnl, nan=0.0, posinf=0.0, neginf=0.0))

        self.logger.info(f"[SIM] TRADE END   {instrument} exit {exit_price:.4f} pnl={pnl:.2f}")

        # ─── Update balance & reasoning ───────────────────────────────
        self.balance = float(np.nan_to_num(self.balance + pnl,
                                           nan=self.position_manager.initial_balance + pnl))
        self.peak_balance = max(self.peak_balance, self.balance)
        self.current_drawdown = max(
            (self.peak_balance - self.balance) / (self.peak_balance + 1e-12),
            0.0
        )

        self.reasoning_trace.extend([
            f"Executed trade for {instrument}:",
            f"    Entry Price: {price:.4f}, Exit Price: {exit_price:.4f}",
            f"    Decision: {'Buy' if sign>0 else 'Sell'}",
            f"    Volatility: {vol:.6f}, Raw Size: {raw_size:.6f} lots, Capped to {size:.6f}"
        ])
        self.logger.info(self.reasoning_trace[-1])
        self.logger.info(
            f"[{instrument}] intensity={intensity:.4f} | "
            f"size={size:.6f} | vol={vol:.6f} | "
            f"dd={self.current_drawdown:.4f} | "
            f"max_pct={self.position_manager.max_pct:.4f}"
        )

        return {
            "instrument":  instrument,
            "pnl":         pnl,
            "duration":    hold_steps,
            "exit_reason": "timeout",
            "size":        size if sign > 0 else -size,
            "features":    np.array([exit_price, pnl, hold_steps], np.float32),
        }
    
    # # ==================================================================
    # #  Live trade execution via MetaTrader5
    # # ==================================================================
    # def _execute_live_trade(self, instrument: str, size: float, intensity: float) -> dict:
    #     """
    #     Execute a single live trade via MetaTrader5.
    #     Assumes:
    #     - `self.live_mode` is True
    #     - `self.logger` is available for logging
    #     - `self.balance` and `self.peak_balance` track account balance
    #     - `self.point_value` is defined (if needed for any conversions)
    #     Returns a trade dict on success, or an empty dict on failure.
    #     """
    #     # Convert symbol and ensure it's selected
    #     symbol = instrument.replace("/", "")
    #     if not mt5.symbol_select(symbol, True):
    #         self.logger.warning(f"Could not select {symbol}")
    #         return {}

    #     info = mt5.symbol_info(symbol)
    #     if info is None:
    #         self.logger.warning(f"No symbol info for {symbol}")
    #         return {}

    #     # Determine volume step & min/max volumes
    #     step = info.volume_step or 0.01
    #     vmin = info.volume_min or step
    #     vmax = info.volume_max or 100.0

    #     # Round down `size` to nearest step
    #     size = math.floor(size / step) * step
    #     if size < vmin:
    #         self.logger.info(f"Signal too small ({size:.3f} < {vmin}) – forcing minimum size")
    #         size = vmin
    #     size = min(size, vmax)

    #     # Get tick data for bid/ask
    #     tick = mt5.symbol_info_tick(symbol)
    #     if tick is None:
    #         self.logger.warning(f"No tick for {symbol}")
    #         return {}

    #     price_live = tick.ask if intensity > 0 else tick.bid

    #     # Build and send the order request
    #     request = {
    #         "action":       mt5.TRADE_ACTION_DEAL,
    #         "symbol":       symbol,
    #         "volume":       size,
    #         "type":         mt5.ORDER_TYPE_BUY if intensity > 0 else mt5.ORDER_TYPE_SELL,
    #         "price":        price_live,
    #         "deviation":    20,
    #         "magic":        202406,
    #         "comment":      "AI live trade",
    #         "type_time":    mt5.ORDER_TIME_GTC,
    #         "type_filling": mt5.ORDER_FILLING_IOC,
    #     }
    #     result = mt5.order_send(request)
    #     if result.retcode != mt5.TRADE_RETCODE_DONE:
    #         self.logger.warning(f"MT5 order failed: {result.comment}")
    #         return {}

    #     # Update balance from account_info
    #     acc = mt5.account_info()
    #     if acc and hasattr(acc, "balance"):
    #         self.balance = float(acc.balance)
    #         self.peak_balance = max(self.peak_balance, self.balance)

    #     self.logger.info(
    #         f"LIVE {'BUY' if intensity > 0 else 'SELL'} {symbol} "
    #         f"{size:.2f} lot @ {price_live}"
    #     )

    #     return {
    #         "instrument":  instrument,
    #         "pnl":         0.0,
    #         "duration":    1,
    #         "exit_reason": "executed",
    #         "size":        size if intensity > 0 else -size,
    #         "features":    np.array([price_live, 0.0, 0.0], np.float32),
    #     }


    # ==================================================================
    #  Genome metrics, helpers, checkpointing — UNCHANGED
    # ==================================================================
    def get_genome_metrics(self) -> Dict[str, float]:
        best_idx = int(np.argmax(self.strategy_pool.fitness))
        genome = self.strategy_pool.population[best_idx]
        sl_base, tp_base, vol_scale, regime_adapt = genome
        return {
            "sl_base":     float(sl_base),
            "tp_base":     float(tp_base),
            "vol_scale":   float(vol_scale),
            "regime_adapt": float(regime_adapt),
            "fitness":     float(self.strategy_pool.fitness[best_idx]),
        }

    def get_recent_bars(self, instrument: str = "EUR/USD", timeframe: str = "D1", n: int = 5) -> Dict[str, Any]:
        df = self.data[instrument][timeframe]
        tail = df.iloc[-n:]
        times = [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in tail.index]
        return {
            "time":       times,
            "close":      tail["close"].tolist(),
            "volatility": tail["volatility"].tolist(),
        }

    def _finish_episode(self, last_pnl: float):
        if not self._ep_pnls:
            return
        balances = np.array(
            [self.initial_balance] +
            list(np.cumsum(self._ep_pnls) + self.initial_balance),
            dtype=np.float32,
        )
        returns = np.diff(balances) / (balances[:-1] + 1e-8)
        returns = np.nan_to_num(returns)
        mean_r = returns.mean()
        std_r  = returns.std() + 1e-8
        sharpe = np.clip(mean_r / std_r * np.sqrt(250), -2.0, 5.0)
        negative = returns[returns < 0]
        std_neg = (negative.std() + 1e-8) if negative.size > 0 else std_r
        sortino = np.clip(mean_r / std_neg * np.sqrt(250), -1.0, 3.0)
        dd = self.current_drawdown

        # Fitness for active genome (could be more sophisticated)
        def eval_fn(genome: np.ndarray) -> float:
            return float(0.4 * last_pnl + 0.3 * sharpe + 0.3 * sortino - 0.5 * dd)

        self.strategy_pool.evaluate_population(eval_fn)
        self.strategy_pool.evolve_strategies()
        # Save/checkpoint all meta/neuro modules if you want (MetaRL, playbook, etc.)
        try:
            self._save_checkpoints()
        except Exception as e:
            self.logger.error(f"Env checkpoint save failed: {e}")

    # ==================================================================
    #  Serialization helpers
    # ==================================================================
    def get_state(self) -> Dict[str, Any]:
        """Return enough to restore a paused episode."""
        return {
            "balance":           self.balance,
            "peak_balance":      self.peak_balance,
            "current_step":      self.current_step,
            "current_drawdown":  self.current_drawdown,
            "position_manager":  self.position_manager.get_state(),
            "open_positions":    self.open_positions,
            "_ep_pnls":          self._ep_pnls,
            "_ep_durations":     self._ep_durations,
            "_ep_drawdowns":     self._ep_drawdowns,
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.balance          = state.get("balance", self.initial_balance)
        self.peak_balance     = state.get("peak_balance", self.balance)
        self.current_step     = state.get("current_step", 0)
        self.current_drawdown = state.get("current_drawdown", 0.0)
        self.open_positions   = state.get("open_positions", [])
        self._ep_pnls         = state.get("_ep_pnls", [])
        self._ep_durations    = state.get("_ep_durations", [])
        self._ep_drawdowns    = state.get("_ep_drawdowns", [])
        # restore nested module
        self.position_manager.set_state(state.get("position_manager", {}))

    def _ckpt(self, name: str) -> str:
        return os.path.join(self.ckpt_dir, name)

    def _save_checkpoints(self):
        try:
            # Save meta-learning model state (meta_rl)
            if hasattr(self.meta_rl, "state_dict"):
                torch.save(self.meta_rl.state_dict(), self._ckpt("meta_rl.pt"))
            else:
                self.logger.info("MetaRLController has no state_dict, skipping its checkpoint save.")

            # Save the strategy arbiter weights
            np.save(self._ckpt("arbiter_w.npy"), self.arbiter.weights)

            # Save the strategy pool data
            with open(self._ckpt("genome_pool.pkl"), "wb") as f:
                pickle.dump({
                    "population": self.strategy_pool.population,
                    "fitness": self.strategy_pool.fitness,
                    "epoch": self.strategy_pool.epoch,
                }, f)

            # Save the state of each module
            module_states = {}
            for module in self.pipeline.modules:
                if hasattr(module, "get_state"):
                    module_states[module.__class__.__name__] = module.get_state()

            # Save the environment's state (PositionManager, MistakeMemory, etc.)
            env_state = {
                "position_manager": self.position_manager.get_state(),
                "mistake_memory": self.mistake_memory.get_state(),
                "balance": self.balance,
                "current_step": self.current_step,
                "current_drawdown": self.current_drawdown,
                "open_positions": self.open_positions,
                "votes_log": self.votes_log,  # Save votes log
                "module_states": module_states  # Save module states here
            }

            with open(self._ckpt("env_state.pkl"), "wb") as f:
                pickle.dump(env_state, f)

            self.logger.info("Checkpoint saved successfully.")
        except Exception as e:
            self.logger.error(f"Env checkpoint save failed: {e}")

    def _maybe_load_checkpoints(self):
        try:
            # Load the meta-learning model state (meta_rl)
            if os.path.isfile(self._ckpt("meta_rl.pt")):
                self.meta_rl.load_state_dict(torch.load(self._ckpt("meta_rl.pt")))

            # Load the strategy arbiter weights
            if os.path.isfile(self._ckpt("arbiter_w.npy")):
                self.arbiter.weights = np.load(self._ckpt("arbiter_w.npy"))

            # Load the strategy pool data
            if os.path.isfile(self._ckpt("genome_pool.pkl")):
                with open(self._ckpt("genome_pool.pkl"), "rb") as f:
                    d = pickle.load(f)
                self.strategy_pool.population = d["population"]
                self.strategy_pool.fitness = d["fitness"]
                self.strategy_pool.epoch = d["epoch"]

            # Load the environment state
            if os.path.isfile(self._ckpt("env_state.pkl")):
                with open(self._ckpt("env_state.pkl"), "rb") as f:
                    env_state = pickle.load(f)
                self.set_state(env_state)  # Set the environment state from the file

                # Load module states
                for module_name, state in env_state["module_states"].items():
                    module = getattr(self, module_name.lower(), None)
                    if module and hasattr(module, "set_state"):
                        module.set_state(state)

                # Load votes log
                self.votes_log = env_state.get("votes_log", [])

            self.logger.info("Checkpoints loaded successfully.")
        except Exception as e:
            self.logger.warning(f"Checkpoint load failed: {e}. Starting fresh.")

    # ==================================================================
    #  Diagnostic helper getters — UNCHANGED
    # ==================================================================
    def get_module_status(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for m in self.pipeline.modules + self.arbiter.members:
            nm = m.__class__.__name__
            conf = 0.0
            if hasattr(m, "confidence"):
                try:
                    conf = float(m.confidence(None))
                except Exception:
                    pass
            out[nm] = {"enabled": self.module_enabled.get(nm, True), "confidence": conf}
        return out

    def get_votes_history(self) -> List[Dict[str, float]]:
        return self.votes_log

    def get_reasoning_trace(self) -> List[str]:
        return self.reasoning_trace

    def get_metrics(self) -> Dict[str, float]:
        wr  = float((np.array(self._ep_pnls) > 0).mean()) if self._ep_pnls else 0.0
        avg = float(np.mean(self._ep_pnls)) if self._ep_pnls else 0.0
        last = self._ep_pnls[-1] if self._ep_pnls else 0.0
        return {
            "balance":  self.balance,
            "win_rate": wr,
            "avg_pnl":  avg,
            "last_pnl": last,
            "drawdown": self.current_drawdown,
        }

    def get_instrument_correlations(self) -> Dict[tuple[str, str], float]:
        corrs: Dict[tuple[str, str], float] = {}
        for i1, i2 in combinations(self.instruments, 2):
            df1 = self.data[i1]["D1"]["close"].pct_change().dropna()
            df2 = self.data[i2]["D1"]["close"].pct_change().dropna()
            arr1, arr2 = df1.iloc[-100:].values, df2.iloc[-100:].values
            if arr1.size == 0 or arr2.size == 0:
                corr = 0.0
            else:
                corr = float(np.corrcoef(arr1, arr2)[0, 1])
            corrs[(i1, i2)] = corr
        return corrs

    def get_volatility_profile(self) -> Dict[str, float]:
        vols: Dict[str, float] = {}
        for inst in self.instruments:
            df = self.data[inst]["D1"]
            vols[inst] = df.iloc[self.current_step]["volatility"] if self.current_step < len(df) else 0.0
        return vols

    def get_trade_exit_reasons(self) -> List[str]:
        return [t.get("exit_reason", "unknown") for t in getattr(self, "trades", [])]

    def get_current_correlation(self) -> float:
        xau = self.data["XAU/USD"]["D1"]["close"].pct_change().dropna()
        eur = self.data["EUR/USD"]["D1"]["close"].pct_change().dropna()
        n = min(len(xau), len(eur))
        if n == 0:
            return 0.0
        return float(np.corrcoef(xau.iloc[:n], eur.iloc[:n])[0, 1])

    def get_neuro_activity(self) -> Dict[str, Any]:
        try:
            return {
                "weights": self.meta_rl.get_weights(),
                "gradients": self.meta_rl.get_gradients(),
                "intuition_vector": self.memory_compressor.intuition_vector.tolist(),
                "attention_weights": self.arbiter.weights.tolist(),
            }
        except Exception as e:
            self.logger.error(f"Neuro activity error: {str(e)}")
            return {}

    # ==================================================================
    #  Render / close
    # ==================================================================
    def render(self, mode="human"):
        print(
            f"Step {self.current_step} | MODE={self.mode_manager.get_mode().upper()} | "
            f"Bal={self.balance:.2f} | DD={self.current_drawdown:.2%}"
        )

    def close(self):
        pass

    # ==================================================================
    #  Module toggle helper
    # ==================================================================
    def set_module_enabled(self, name: str, en: bool):
        if name not in self.module_enabled:
            raise KeyError(f"No such module {name}")
        self.module_enabled[name] = en
