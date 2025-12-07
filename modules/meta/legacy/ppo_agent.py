# #!/usr/bin/env python3
# """
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  DEPRECATED - This module has been superseded by the 3-layer architecture   ║
# ║                                                                              ║
# ║  Use instead:                                                                ║
# ║    - modules/meta/ppo_agent_shell.py  (SmartInfoBus gateway)                ║
# ║    - modules/meta/arbiter_logic.py    (Domain logic)                        ║
# ║    - modules/meta/ppo_core.py         (Pure RL engine)                      ║
# ║                                                                              ║
# ║  This file is preserved for reference only. Do not use in new code.         ║
# ║  Deprecated: 2025-12-06                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

# Enhanced PPO Agent / Arbiter - Live + Training (LEGACY)
# =======================================================

# This module implements a production-ready PPO agent that acts as an
# INTELLIGENT ARBITER on top of a committee of experts.

# Key design decisions:

# - SINGLE observation schema (PPOObservationBuilder v2.0, 48 dims)
#   used consistently in:
#     - Training environment (ModernTradingEnv)
#     - Live arbiter (PPOAgent.make_final_decision)

# - PPO action space:
#     - action[0] = trust_score   (-inf..inf, interpreted as how much to trust committee)
#     - action[1] = size_score    (-inf..inf, mapped to position size 0..1)

# - PPOAgent is NOT a voting member; it is the FINAL ARBITER.
#   It consumes:
#     - Expert voting signals
#     - Committee consensus
#     - Risk/memory signals
#     - Unified observation vector

# - When debug mode is enabled (PPOConfig.debug = True),
#   extensive debug logs are emitted via RotatingLogger.
# """

# import asyncio
# import time
# import threading
# from collections import deque, defaultdict
# from dataclasses import dataclass, field, asdict
# from datetime import datetime
# from typing import Dict, Any, List, Optional, Tuple, Union

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim

# from modules.contracts import module_args
# from modules.core.module_base import BaseModule, module
# from modules.core.mixins import (
#     SmartInfoBusTradingMixin,
#     SmartInfoBusRiskMixin,
#     SmartInfoBusStateMixin,
# )
# from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
# from modules.monitoring.performance_tracker import PerformanceTracker
# from modules.utils.audit_utils import RotatingLogger, format_operator_message
# from modules.utils.info_bus import InfoBusManager
# from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
# from modules.voting.core.per_instrument import (
#     PerInstrumentVote,
#     InstrumentProposal,
#     DEFAULT_INSTRUMENTS,
#     extract_instrument_data,
#     analyze_instrument_trend,
# )

# # Unified PPO observation builder (v2.0)
# from modules.meta.ppo_observation_builder import (
#     PPOObservationBuilder,
#     PPOObservationConfig,
#     PPO_OBS_SIZE,
#     PPO_OBS_VERSION,
#     build_ppo_observation,
#     get_ppo_observation_builder,
# )


# # ═══════════════════════════════════════════════════════════════════
# # PPO CONFIGURATION
# # ═══════════════════════════════════════════════════════════════════

# @dataclass
# class PPOConfig:
#     """
#     Configuration for PPO Agent (v2.0)

#     IMPORTANT: obs_size must match PPO_OBS_SIZE from ppo_observation_builder.
#     This ensures training and live inference use identical observation schemas.

#     Version History:
#     - v1.0: obs_size=10 (legacy arbiter-only features)
#     - v2.0: obs_size=48 (unified M15-primary + voting + committee + risk + account)
#     """

#     # Observation schema version - must match ppo_observation_builder
#     obs_version: str = PPO_OBS_VERSION  # "2.0"

#     # Network dimensions - aligned with unified observation builder
#     obs_size: int = PPO_OBS_SIZE  # 48 dims by default
#     act_size: int = 2             # (trust_score, position_size_score)
#     hidden_size: int = 128        # Increased to handle richer observation

#     learning_rate: float = 3e-4
#     device: str = "cpu"

#     # PPO hyperparameters
#     clip_eps: float = 0.2
#     value_coeff: float = 0.5
#     entropy_coeff: float = 0.01
#     gae_lambda: float = 0.95
#     gamma: float = 0.99
#     max_grad_norm: float = 0.5
#     ppo_epochs: int = 4

#     # NOTE: Here `batch_size` is the minimum number of samples before we trigger
#     # a policy update. Full-batch PPO update is used (no mini-batching inside).
#     batch_size: int = 64

#     # Performance thresholds
#     max_processing_time_ms: float = 500
#     circuit_breaker_threshold: int = 3
#     min_performance_score: float = 0.3

#     # Training parameters
#     buffer_size: int = 2048          # Reserved for future full-trajectory updates
#     early_stopping_patience: int = 100
#     lr_decay_patience: int = 50

#     # Debug / logging
#     debug: bool = False              # When True, emit detailed debug logs

#     def __post_init__(self):
#         """Validate configuration consistency."""
#         if self.obs_size != PPO_OBS_SIZE:
#             import warnings
#             warnings.warn(
#                 f"PPOConfig.obs_size ({self.obs_size}) != PPO_OBS_SIZE ({PPO_OBS_SIZE}). "
#                 f"This may cause training/live mismatch. Consider using PPO_OBS_SIZE.",
#                 UserWarning,
#             )


# # ═══════════════════════════════════════════════════════════════════
# # NEURAL NETWORK
# # ═══════════════════════════════════════════════════════════════════

# class EnhancedPPONetwork(nn.Module):
#     """Enhanced PPO network with improved architecture."""

#     def __init__(self, obs_size: int, act_size: int, hidden_size: int = 64):
#         super().__init__()

#         # Shared feature extractor
#         self.feature_extractor = nn.Sequential(
#             nn.Linear(obs_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#         )

#         # Policy head (actor)
#         self.policy_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, act_size * 2),  # mean and log_std
#         )

#         # Value head (critic)
#         self.value_head = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_size // 2, 1),
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         """Initialize network weights."""
#         for mod in self.modules():
#             if isinstance(mod, nn.Linear):
#                 nn.init.orthogonal_(mod.weight, gain=np.sqrt(2))
#                 nn.init.zeros_(mod.bias)

#         # Special initialization for policy output (last linear of policy_head)
#         last = self.policy_head[-1]
#         if isinstance(last, nn.Linear):
#             nn.init.orthogonal_(last.weight, gain=1.0)

#     def forward(self, obs: torch.Tensor):
#         """Forward pass.

#         Returns:
#             action_mean: (batch, act_size)
#             action_log_std: (batch, act_size)
#             value: (batch, 1)
#         """
#         features = self.feature_extractor(obs)

#         # Policy output
#         policy_out = self.policy_head(features)
#         half = policy_out.size(-1) // 2
#         action_mean = policy_out[..., :half]
#         action_log_std = policy_out[..., half:]
#         action_log_std = torch.clamp(action_log_std, -20, 2)

#         # Value output
#         value = self.value_head(features)

#         return action_mean, action_log_std, value


# # ═══════════════════════════════════════════════════════════════════
# # PPO AGENT / ARBITER
# # ═══════════════════════════════════════════════════════════════════

# @module(
#     **module_args(
#         "PPOAgent",
#         description="Intelligent PPO arbiter - consumes committee consensus and makes final GO/NO-GO trading decisions",
#         error_handling=True,
#         hot_reload=True,
#         timeout_ms=3000,
#         # PPOAgent is now the INTELLIGENT ARBITER, not a voter
#         is_voting_member=False,
#         is_final_arbiter=True,
#     )
# )
# class PPOAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
#     """
#     Intelligent PPO Arbiter with SmartInfoBus integration.

#     ROLE: Final decision-maker that consumes:
#     - Committee consensus (aggregated expert votes)
#     - Individual expert signals (for override decisions)
#     - Risk signals (portfolio risk, fragility)
#     - Memory signals (danger zones, patterns)

#     Outputs (on SmartInfoBus):
#     - ppo_final_decision: The final trading decision (direction, confidence, reasoning, gate_passed, position_size)
#     - ppo_gate_passed: Whether to execute the trade (GO/NO-GO)
#     - ppo_position_size: Recommended position size based on confidence
#     """

#     # typed members for Pylance
#     _cfg: PPOConfig
#     device: torch.device
#     network: EnhancedPPONetwork

#     def __init__(
#         self,
#         config: Optional[Union[PPOConfig, Dict[str, Any]]] = None,
#         genome: Optional[Dict[str, Any]] = None,
#         **kwargs,
#     ):
#         # Normalize config to typed dataclass and keep dict version for BaseModule
#         if isinstance(config, dict):
#             custom_config = PPOConfig(
#                 **{k: v for k, v in config.items() if k in PPOConfig.__dataclass_fields__}
#             )
#         else:
#             custom_config = config or PPOConfig()
#         self._cfg = custom_config

#         # Initialize BaseModule with a plain dict config
#         super().__init__(config=asdict(self._cfg), **kwargs)

#         # Initialize advanced systems (logger, bus, trackers, etc.)
#         self._initialize_advanced_systems()

#         # Initialize genome parameters
#         self._initialize_genome_parameters(genome)

#         # Initialize PPO state + obs builder
#         self._initialize_ppo_state()

#         # Initialize neural components
#         self._initialize_neural_components()

#         self.logger.info(
#             format_operator_message(
#                 "[BOT]",
#                 "PPO_AGENT_INITIALIZED",
#                 details=f"Obs: {self._cfg.obs_size}, Actions: {self._cfg.act_size}, Hidden: {self._cfg.hidden_size}",
#                 result="PPO agent ready for training and live arbitration",
#                 context="ppo_initialization",
#             )
#         )

#         # Start monitoring after all initialization is complete
#         self._start_monitoring()

#     # ─────────────────────────────────────────────────────────────
#     # Initialization helpers
#     # ─────────────────────────────────────────────────────────────

#     def _initialize_advanced_systems(self):
#         """Initialize advanced systems for PPO agent."""
#         self.smart_bus = InfoBusManager.get_instance()
#         self.logger = RotatingLogger(
#             name="PPOAgent",
#             log_path="logs/meta/ppo_agent.log",
#             max_lines=5000,
#             operator_mode=True,
#             plain_english=True,
#         )

#         # Debug flag from config
#         self.debug: bool = bool(getattr(self._cfg, "debug", False))
#         if self.debug:
#             self.logger.info("[DEBUG] PPOAgent debug logging ENABLED")

#         self.error_pinpointer = ErrorPinpointer()
#         self.error_handler = create_error_handler("PPOAgent", self.error_pinpointer)
#         self.english_explainer = EnglishExplainer()
#         self.system_utilities = SystemUtilities()
#         self.performance_tracker = PerformanceTracker()

#         # Circuit breaker for neural operations
#         self.circuit_breaker = {
#             "failures": 0,
#             "last_failure": 0.0,
#             "state": "CLOSED",
#             "threshold": self._cfg.circuit_breaker_threshold,
#         }

#         # Health monitoring
#         self._health_status = "healthy"
#         self._last_health_check = time.time()
#         # Note: _start_monitoring() is called at the end of __init__

#     def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
#         """Initialize genome-based parameters into _cfg (typed)."""
#         if genome:
#             self.genome = {
#                 "obs_size": int(genome.get("obs_size", self._cfg.obs_size)),
#                 "act_size": int(genome.get("act_size", self._cfg.act_size)),
#                 "hidden_size": int(genome.get("hidden_size", self._cfg.hidden_size)),
#                 "learning_rate": float(genome.get("learning_rate", self._cfg.learning_rate)),
#                 "clip_eps": float(genome.get("clip_eps", self._cfg.clip_eps)),
#                 "value_coeff": float(genome.get("value_coeff", self._cfg.value_coeff)),
#                 "entropy_coeff": float(genome.get("entropy_coeff", self._cfg.entropy_coeff)),
#                 "gae_lambda": float(genome.get("gae_lambda", self._cfg.gae_lambda)),
#                 "gamma": float(genome.get("gamma", self._cfg.gamma)),
#                 "device": str(genome.get("device", self._cfg.device)),
#             }
#             # Apply to _cfg
#             for key, value in self.genome.items():
#                 if key in PPOConfig.__dataclass_fields__:
#                     setattr(self._cfg, key, value)
#         else:
#             self.genome = {
#                 "obs_size": self._cfg.obs_size,
#                 "act_size": self._cfg.act_size,
#                 "hidden_size": self._cfg.hidden_size,
#                 "learning_rate": self._cfg.learning_rate,
#                 "clip_eps": self._cfg.clip_eps,
#                 "value_coeff": self._cfg.value_coeff,
#                 "entropy_coeff": self._cfg.entropy_coeff,
#                 "gae_lambda": self._cfg.gae_lambda,
#                 "gamma": self._cfg.gamma,
#                 "device": self._cfg.device,
#             }

#     def _initialize_ppo_state(self):
#         """Initialize PPO-specific state."""
#         # Device setup
#         self.device = torch.device(self._cfg.device)

#         # Unified observation builder (v2.0) - same as used in training env
#         self.obs_builder: PPOObservationBuilder = get_ppo_observation_builder()
#         self.logger.info(
#             f"[PPO] Observation builder v{self.obs_builder.version} initialized "
#             f"(obs_size={self.obs_builder.obs_size})"
#         )

#         # Experience buffer
#         self.buffer: Dict[str, List[Any]] = {
#             "observations": [],
#             "actions": [],
#             "log_probs": [],
#             "values": [],
#             "rewards": [],
#             "advantages": [],
#             "returns": [],
#             "dones": [],
#         }

#         # Performance tracking
#         self.episode_rewards = deque(maxlen=100)
#         self.episode_lengths = deque(maxlen=100)
#         self.policy_losses = deque(maxlen=100)
#         self.value_losses = deque(maxlen=100)
#         self.entropy_losses = deque(maxlen=100)

#         # Training statistics
#         self.training_stats: Dict[str, Any] = {
#             "total_updates": 0,
#             "episodes_completed": 0,
#             "best_episode_reward": -np.inf,
#             "avg_episode_reward": 0.0,
#             "policy_loss_trend": 0.0,
#             "value_loss_trend": 0.0,
#             "entropy_trend": 0.0,
#             "gradient_norm": 0.0,
#             "explained_variance": 0.0,
#             "learning_rate": self._cfg.learning_rate,
#         }

#         # Action tracking
#         self.last_action = np.zeros(self._cfg.act_size, dtype=np.float32)
#         self.action_history = deque(maxlen=1000)
#         self.action_statistics = {
#             "mean_action": np.zeros(self._cfg.act_size),
#             "action_std": np.ones(self._cfg.act_size),
#             "action_range": np.ones(self._cfg.act_size),
#             "exploration_level": 0.5,
#         }

#         # Market context integration
#         self.market_context_history = deque(maxlen=50)
#         self.context_performance = defaultdict(lambda: {"rewards": [], "count": 0})

#         # Learning adaptation
#         self.early_stopping_counter = 0
#         self.best_performance = -np.inf
#         self.performance_plateau_counter = 0

#         # Recent IO snapshots for bus publications
#         self._last_obs_vec: Optional[np.ndarray] = None
#         self._last_action_std: Optional[List[float]] = None
#         self._recent_rewards = deque(maxlen=100)

#         # Voting direction hysteresis to prevent flip-flopping
#         self._last_direction: str = "flat"  # Tracks last emitted direction for hysteresis
#         self._direction_hold_count: int = 0  # How many ticks we've held this direction

#         # Neural performance metrics
#         self._neural_performance = {
#             "forward_passes": 0,
#             "backward_passes": 0,
#             "average_loss": 0.0,
#             "gradient_stability": 1.0,
#         }

#     def _initialize_neural_components(self):
#         """Initialize neural network components."""
#         try:
#             # Main network
#             self.network = EnhancedPPONetwork(
#                 self._cfg.obs_size,
#                 self._cfg.act_size,
#                 self._cfg.hidden_size,
#             ).to(self.device)

#             # Optimizer with improved settings
#             self.optimizer = optim.Adam(
#                 self.network.parameters(),
#                 lr=self._cfg.learning_rate,
#                 eps=1e-5,
#                 weight_decay=1e-4,
#             )

#             # Learning rate scheduler
#             self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#                 self.optimizer,
#                 mode="max",
#                 factor=0.8,
#                 patience=self._cfg.lr_decay_patience,
#             )

#             self.logger.info("Neural components initialized successfully")
#             if self.debug:
#                 total_params = sum(p.numel() for p in self.network.parameters())
#                 self.logger.debug(
#                     f"[DEBUG][PPO] Network initialized "
#                     f"(obs_size={self._cfg.obs_size}, act_size={self._cfg.act_size}, "
#                     f"hidden_size={self._cfg.hidden_size}, params={total_params})"
#                 )

#         except Exception as e:
#             self.logger.error(f"Neural component initialization failed: {e}")
#             self._health_status = "error"

#     def _start_monitoring(self):
#         """Start background monitoring."""

#         def monitoring_loop():
#             while getattr(self, "_monitoring_active", True):
#                 try:
#                     self._update_ppo_health()
#                     self._analyze_learning_progress()
#                     time.sleep(30)
#                 except Exception as e:
#                     self.logger.error(f"Monitoring error: {e}")

#         self._monitoring_active = True
#         monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
#         monitor_thread.start()

#     def _initialize(self):
#         """
#         Initialize module (called by BaseModule early). Safe even if advanced systems
#         aren't ready yet — we simply defer until __init__ finishes.
#         """
#         try:
#             if not hasattr(self, "smart_bus"):
#                 # Defer silently to avoid noisy logs; we'll publish on the next cycle.
#                 return

#             initial_status = {
#                 "episodes_completed": 0,
#                 "training_updates": 0,
#                 "average_reward": 0.0,
#                 "learning_rate": float(getattr(self._cfg, "learning_rate", 3e-4)),
#                 "performance_score": 0.0,
#             }

#             self.smart_bus.set(
#                 "agent_performance",
#                 initial_status,
#                 module="PPOAgent",
#                 thesis="Initial PPO agent performance status",
#             )

#         except Exception as e:
#             if hasattr(self, "logger"):
#                 self.logger.error(f"Initialization failed: {e}")
#             else:
#                 print(f"[PPOAgent] Initialization failed (pre-logger): {e}")

#     # ─────────────────────────────────────────────────────────────
#     # Main process entrypoint
#     # ─────────────────────────────────────────────────────────────

#     async def process(self, **inputs) -> Dict[str, Any]:
#         """Process PPO agent operations (training + arbitration)."""
#         start_time = time.time()

#         try:
#             # Extract PPO data
#             ppo_data = await self._extract_ppo_data(**inputs)

#             if not ppo_data:
#                 return await self._handle_no_data_fallback()

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] process() inputs received: "
#                     f"keys={list(inputs.keys())}, "
#                     f"has_observation={'observation' in inputs}, "
#                     f"has_experience={'experience' in inputs}"
#                 )

#             # Process action selection if observation provided
#             action_result: Dict[str, Any] = {}
#             if "observation" in ppo_data:
#                 action_result = await self._process_action_selection(ppo_data)

#             # Process training if experience provided
#             training_result: Dict[str, Any] = {}
#             if "experience" in ppo_data:
#                 training_result = await self._process_training(ppo_data)

#             # Update agent metrics
#             metrics_result = await self._update_agent_metrics()

#             # Combine results
#             result: Dict[str, Any] = {**action_result, **training_result, **metrics_result}

#             # Generate thesis (performance-oriented)
#             thesis = await self._generate_ppo_thesis(ppo_data, result)
#             result["_thesis"] = thesis

#             # ═══════════════════════════════════════════════════════════════════
#             # INTELLIGENT ARBITER: Make final GO/NO-GO decision
#             # ═══════════════════════════════════════════════════════════════════
#             arbiter_result = await self.make_final_decision(
#                 observation=ppo_data.get("observation"), **inputs
#             )
#             result.update(arbiter_result)

#             # Update thesis with arbiter decision reasoning
#             arbiter_decision = arbiter_result.get("ppo_final_decision", {})
#             thesis = f"{thesis} | ARBITER: {arbiter_decision.get('reasoning', 'N/A')}"
#             result["_thesis"] = thesis

#             # --- Legacy voting outputs (for backward compatibility) ---
#             action_vec = result.get("action", None)
#             vote_payload = await self.vote(
#                 observation=ppo_data.get("observation"),
#                 action_vec=action_vec,
#                 market_data=ppo_data.get("market_data"),
#                 thesis=thesis,
#             )
#             result["PPOAgent_voting_proposal"] = vote_payload
#             result["PPOAgent_confidence"] = (
#                 float(vote_payload.get("confidence", 0.0))
#                 if isinstance(vote_payload, dict)
#                 else 0.0
#             )

#             # Update SmartInfoBus (including arbiter keys)
#             await self._update_ppo_smart_bus(result, thesis)

#             # Ensure required outputs are always present
#             self._ensure_required_outputs(result)

#             # Record success
#             processing_time = (time.time() - start_time) * 1000.0
#             self._record_success(processing_time)

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] process() completed in {processing_time:.2f} ms, "
#                     f"gate_passed={result.get('ppo_gate_passed')}, "
#                     f"position_size={result.get('ppo_position_size')}"
#                 )

#             return result

#         except Exception as e:
#             return await self._handle_ppo_error(e, start_time)

#     # ─────────────────────────────────────────────────────────────
#     # Data extraction / normalization
#     # ─────────────────────────────────────────────────────────────

#     async def _extract_ppo_data(self, **inputs) -> Optional[Dict[str, Any]]:
#         """Extract PPO data from SmartInfoBus and direct inputs."""
#         try:
#             # Observations snapshot (from last forward pass if available)
#             observations = None
#             try:
#                 if isinstance(self._last_obs_vec, np.ndarray):
#                     observations = self._last_obs_vec.astype(float).tolist()
#                 elif isinstance(self._last_obs_vec, list):
#                     observations = self._last_obs_vec
#             except Exception:
#                 observations = None

#             # Recent rewards snapshot from internal deque
#             rewards = list(self._recent_rewards)[-10:] if self._recent_rewards else []

#             # Market data can legitimately come from the bus
#             market_data = self.smart_bus.get("market_data", "PPOAgent") or {}

#             # Training signals derived from current training stats (avoid bus get)
#             training_signals = {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get("explained_variance", 0.0),
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#             }

#             # Direct inputs
#             observation = inputs.get("observation")
#             experience = inputs.get("experience")

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] _extract_ppo_data: "
#                     f"obs_from_input={observation is not None}, "
#                     f"experience_present={experience is not None}"
#                 )

#             return {
#                 "observations": observations,
#                 "rewards": rewards,
#                 "market_data": market_data,
#                 "training_signals": training_signals,
#                 "observation": observation,
#                 "experience": experience,
#                 "timestamp": datetime.now().isoformat(),
#             }

#         except Exception as e:
#             self.logger.error(f"Failed to extract PPO data: {e}")
#             return None

#     def _normalize_observation(self, observation: Any) -> np.ndarray:
#         """
#         Normalize raw observation into a fixed-length float32 vector.

#         v2.0 behavior:
#         - If observation is None, builds a full observation from SmartInfoBus
#           using the unified PPOObservationBuilder (48 dims by default).
#         - If observation is already the correct size (obs_size), returns as-is
#           after sanitizing NaN/Inf.
#         - Otherwise, pads/truncates to obs_size.
#         """
#         obs_size = int(getattr(self._cfg, "obs_size", PPO_OBS_SIZE) or PPO_OBS_SIZE)

#         try:
#             # If None, use the unified observation builder to fetch from bus
#             if observation is None:
#                 if self.debug:
#                     self.logger.debug(
#                         "[DEBUG][PPO] _normalize_observation: observation=None, "
#                         "building from SmartInfoBus via PPOObservationBuilder"
#                     )
#                 obs_vec = self.obs_builder.build(
#                     smart_bus=self.smart_bus, module_name="PPOAgent"
#                 )
#                 return obs_vec.astype(np.float32, copy=False)

#             # If dict, treat as an ad-hoc numeric feature dict
#             if isinstance(observation, dict):
#                 keys = [
#                     k
#                     for k in observation.keys()
#                     if isinstance(observation[k], (int, float, np.number))
#                 ]
#                 arr = (
#                     np.array([float(observation[k]) for k in keys], dtype=np.float32)
#                     if keys
#                     else np.array([], dtype=np.float32)
#                 )
#             else:
#                 # Generic array-like
#                 arr = np.array(observation, dtype=np.float32).reshape(-1)

#             # Sanitize NaN/Inf
#             arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

#             # Pad or truncate to obs_size
#             if arr.size == obs_size:
#                 out = arr.astype(np.float32, copy=False)
#             elif arr.size < obs_size:
#                 out = np.zeros(obs_size, dtype=np.float32)
#                 out[: arr.size] = arr
#             else:  # arr.size > obs_size
#                 out = arr[:obs_size]

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] _normalize_observation: "
#                     f"input_type={type(observation).__name__}, "
#                     f"raw_size={getattr(arr, 'size', None)}, "
#                     f"final_size={out.size}, "
#                     f"min={float(np.min(out)):.4f}, max={float(np.max(out)):.4f}"
#                 )

#             return out

#         except Exception as e:
#             self.logger.warning(
#                 f"[PPO] Observation normalization failed: {e}, using zeros"
#             )
#             return np.zeros(obs_size, dtype=np.float32)

#     def build_unified_observation(self) -> np.ndarray:
#         """
#         Build a unified observation using the PPO observation builder.

#         This method fetches all required data from SmartInfoBus and constructs
#         the canonical observation vector (obs_size dims) used for both training
#         and live arbitration.
#         """
#         try:
#             obs_vec = self.obs_builder.build(
#                 smart_bus=self.smart_bus, module_name="PPOAgent"
#             )
#             obs_vec = np.asarray(obs_vec, dtype=np.float32).reshape(-1)
#             if obs_vec.size != self._cfg.obs_size:
#                 # Pad/truncate defensively
#                 out = np.zeros(self._cfg.obs_size, dtype=np.float32)
#                 out[: min(self._cfg.obs_size, obs_vec.size)] = obs_vec[: self._cfg.obs_size]
#                 obs_vec = out
#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] build_unified_observation: size={obs_vec.size}, "
#                     f"min={float(np.min(obs_vec)):.4f}, max={float(np.max(obs_vec)):.4f}"
#                 )
#             return obs_vec
#         except Exception as e:
#             self.logger.warning(
#                 f"[PPO] build_unified_observation failed: {e}, returning zeros"
#             )
#             return np.zeros(self._cfg.obs_size, dtype=np.float32)

#     # ─────────────────────────────────────────────────────────────
#     # Action selection & training
#     # ─────────────────────────────────────────────────────────────

#     async def _process_action_selection(self, ppo_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process action selection (forward pass) from an observation."""
#         try:
#             observation = ppo_data.get("observation")

#             # Normalize observation to a fixed-length numeric vector
#             obs_vec = self._normalize_observation(observation)
#             obs_tensor = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] _process_action_selection: obs_shape={obs_tensor.shape}"
#                 )

#             # Forward pass
#             with torch.no_grad():
#                 action_mean, action_log_std, value = self.network(obs_tensor)

#                 # Create action distribution
#                 action_std = torch.exp(action_log_std)
#                 try:
#                     self._last_action_std = (
#                         action_std.squeeze().cpu().numpy().tolist()
#                     )
#                 except Exception:
#                     self._last_action_std = None

#                 dist = torch.distributions.Normal(action_mean, action_std)

#                 # Sample action
#                 action = dist.sample()
#                 log_prob = dist.log_prob(action).sum(dim=-1)

#                 # Convert to numpy
#                 action_np = action.squeeze().cpu().numpy()
#                 log_prob_np = float(log_prob.item())
#                 value_np = float(value.squeeze().cpu().item())

#             # Update action tracking
#             self.last_action = action_np
#             self.action_history.append(action_np.copy())
#             try:
#                 self._last_obs_vec = obs_vec.copy()
#             except Exception:
#                 self._last_obs_vec = None
#             self._update_action_statistics()

#             # Keep last log_prob/value so record_step has them
#             self._last_log_prob = log_prob_np
#             self._last_value = value_np

#             # Update neural performance
#             self._neural_performance["forward_passes"] += 1

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] Action selection: action={action_np.tolist()}, "
#                     f"log_prob={log_prob_np:.4f}, value_est={value_np:.4f}"
#                 )

#             return {
#                 "action_selected": True,
#                 "action": action_np.tolist(),
#                 "log_prob": log_prob_np,
#                 "value_estimate": value_np,
#                 "action_std": action_std.squeeze().cpu().numpy().tolist(),
#             }

#         except Exception as e:
#             self.logger.error(f"Action selection failed: {e}")
#             return {"action_selected": False, "error": str(e)}

#     async def _process_training(self, ppo_data: Dict[str, Any]) -> Dict[str, Any]:
#         """Process PPO training step, if enough experience is available."""
#         try:
#             experience = ppo_data["experience"]

#             # Add experience to buffer
#             if isinstance(experience, dict):
#                 for key in ["observation", "action", "reward", "log_prob", "value", "done"]:
#                     if key in experience:
#                         if key == "observation":
#                             self.buffer["observations"].append(experience[key])
#                         elif key == "action":
#                             self.buffer["actions"].append(experience[key])
#                         elif key == "reward":
#                             self.buffer["rewards"].append(experience[key])
#                             try:
#                                 self._recent_rewards.append(float(experience[key]))
#                             except Exception:
#                                 pass
#                         elif key == "log_prob":
#                             self.buffer["log_probs"].append(experience[key])
#                         elif key == "value":
#                             self.buffer["values"].append(experience[key])
#                         elif key == "done":
#                             self.buffer["dones"].append(experience[key])

#             current_size = len(self.buffer["observations"])

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] _process_training: buffer_size={current_size}, "
#                     f"batch_size_threshold={self._cfg.batch_size}"
#                 )

#             # Check if buffer is ready for training
#             if current_size >= self._cfg.batch_size:
#                 update_result = await self._perform_policy_update()
#                 return {"training_performed": True, "update_result": update_result}
#             else:
#                 return {"training_performed": False, "buffer_size": current_size}

#         except Exception as e:
#             self.logger.error(f"Training failed: {e}")
#             return {"training_performed": False, "error": str(e)}

#     async def _perform_policy_update(self) -> Dict[str, Any]:
#         """Perform a PPO policy update using the accumulated buffer."""
#         try:
#             # Compute advantages and returns
#             self._compute_gae_returns()

#             # Convert buffer to tensors
#             observations = torch.tensor(
#                 np.array(self.buffer["observations"]), dtype=torch.float32
#             ).to(self.device)
#             actions = torch.tensor(
#                 np.array(self.buffer["actions"]), dtype=torch.float32
#             ).to(self.device)
#             old_log_probs = torch.tensor(
#                 np.array(self.buffer["log_probs"]), dtype=torch.float32
#             ).to(self.device)
#             advantages = torch.tensor(
#                 np.array(self.buffer["advantages"]), dtype=torch.float32
#             ).to(self.device)
#             returns = torch.tensor(
#                 np.array(self.buffer["returns"]), dtype=torch.float32
#             ).to(self.device)

#             # Normalize advantages
#             advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#             total_policy_loss = 0.0
#             total_value_loss = 0.0
#             total_entropy_loss = 0.0
#             grad_norm: float = 0.0
#             values = None

#             for _ in range(self._cfg.ppo_epochs):
#                 # Forward pass
#                 action_mean, action_log_std, values = self.network(observations)

#                 # Create distribution
#                 action_std = torch.exp(action_log_std)
#                 dist = torch.distributions.Normal(action_mean, action_std)

#                 # Calculate new log probs and entropy
#                 new_log_probs = dist.log_prob(actions).sum(dim=-1)
#                 entropy = dist.entropy().sum(dim=-1)

#                 # Calculate ratio and clipped surrogate loss
#                 ratio = torch.exp(new_log_probs - old_log_probs)
#                 surr1 = ratio * advantages
#                 surr2 = torch.clamp(
#                     ratio, 1 - self._cfg.clip_eps, 1 + self._cfg.clip_eps
#                 ) * advantages
#                 policy_loss = -torch.min(surr1, surr2).mean()

#                 # Value loss
#                 value_loss = F.mse_loss(values.squeeze(), returns)

#                 # Entropy loss
#                 entropy_loss = -entropy.mean()

#                 # Total loss
#                 loss = (
#                     policy_loss
#                     + self._cfg.value_coeff * value_loss
#                     + self._cfg.entropy_coeff * entropy_loss
#                 )

#                 # Backward pass
#                 self.optimizer.zero_grad()
#                 loss.backward()

#                 # Gradient clipping
#                 grad_norm_t = torch.nn.utils.clip_grad_norm_(
#                     self.network.parameters(), self._cfg.max_grad_norm
#                 )
#                 grad_norm = float(grad_norm_t)

#                 self.optimizer.step()

#                 total_policy_loss += float(policy_loss.item())
#                 total_value_loss += float(value_loss.item())
#                 total_entropy_loss += float(entropy_loss.item())

#             # Update statistics
#             avg_policy_loss = total_policy_loss / self._cfg.ppo_epochs
#             avg_value_loss = total_value_loss / self._cfg.ppo_epochs
#             avg_entropy_loss = total_entropy_loss / self._cfg.ppo_epochs

#             self.policy_losses.append(avg_policy_loss)
#             self.value_losses.append(avg_value_loss)
#             self.entropy_losses.append(avg_entropy_loss)

#             self.training_stats["total_updates"] += 1
#             self.training_stats["policy_loss_trend"] = avg_policy_loss
#             self.training_stats["value_loss_trend"] = avg_value_loss
#             self.training_stats["entropy_trend"] = avg_entropy_loss
#             self.training_stats["gradient_norm"] = float(grad_norm)

#             # Explained variance
#             with torch.no_grad():
#                 if values is not None:
#                     explained_var = 1 - torch.var(returns - values.squeeze()) / torch.var(
#                         returns
#                     )
#                     self.training_stats["explained_variance"] = float(explained_var)
#                 else:
#                     explained_var = torch.tensor(0.0)
#                     self.training_stats["explained_variance"] = 0.0

#             # Update neural performance
#             self._neural_performance["backward_passes"] += 1
#             self._neural_performance["average_loss"] = avg_policy_loss + avg_value_loss

#             if self.debug:
#                 self.logger.debug(
#                     "[DEBUG][PPO] Policy update completed: "
#                     f"policy_loss={avg_policy_loss:.4f}, "
#                     f"value_loss={avg_value_loss:.4f}, "
#                     f"entropy_loss={avg_entropy_loss:.4f}, "
#                     f"grad_norm={grad_norm:.4f}, "
#                     f"explained_var={float(explained_var):.4f}"
#                 )

#             # Clear buffer
#             self._clear_buffer()

#             return {
#                 "policy_loss": avg_policy_loss,
#                 "value_loss": avg_value_loss,
#                 "entropy_loss": avg_entropy_loss,
#                 "gradient_norm": float(grad_norm),
#                 "explained_variance": float(explained_var),
#                 "epochs_completed": self._cfg.ppo_epochs,
#             }

#         except Exception as e:
#             self.logger.error(f"Policy update failed: {e}")
#             return {"error": str(e)}

#     def _compute_gae_returns(self):
#         """Compute GAE advantages and returns."""
#         rewards = np.array(self.buffer["rewards"], dtype=np.float32)
#         values = np.array(self.buffer["values"], dtype=np.float32)
#         dones = np.array(self.buffer["dones"], dtype=np.float32)

#         advantages = np.zeros_like(rewards, dtype=np.float32)
#         returns = np.zeros_like(rewards, dtype=np.float32)

#         last_gae = 0.0

#         for t in reversed(range(len(rewards))):
#             if t == len(rewards) - 1:
#                 next_value = 0.0
#                 next_non_terminal = 1.0 - dones[t]
#             else:
#                 next_value = values[t + 1]
#                 next_non_terminal = 1.0 - dones[t]

#             delta = rewards[t] + self._cfg.gamma * next_value * next_non_terminal - values[t]
#             last_gae = (
#                 delta
#                 + self._cfg.gamma
#                 * self._cfg.gae_lambda
#                 * next_non_terminal
#                 * last_gae
#             )
#             advantages[t] = last_gae

#         returns = advantages + values

#         self.buffer["advantages"] = advantages.tolist()
#         self.buffer["returns"] = returns.tolist()

#     def _update_action_statistics(self):
#         """Update action statistics from recent history."""
#         if len(self.action_history) > 10:
#             actions = np.array(list(self.action_history)[-100:])  # Last 100 actions

#             self.action_statistics["mean_action"] = np.mean(actions, axis=0)
#             self.action_statistics["action_std"] = np.std(actions, axis=0)
#             self.action_statistics["action_range"] = np.ptp(actions, axis=0)

#             # Exploration level (simple entropy proxy)
#             action_std = self.action_statistics["action_std"]
#             action_entropy = -np.sum(
#                 action_std * np.log(action_std + 1e-8)
#             )  # not a true entropy but monotonic
#             self.action_statistics["exploration_level"] = float(
#                 np.clip(action_entropy / self._cfg.act_size, 0.0, 1.0)
#             )

#     def _clear_buffer(self):
#         """Clear experience buffer."""
#         for key in self.buffer:
#             self.buffer[key].clear()

#     # ─────────────────────────────────────────────────────────────
#     # Agent metrics / thesis
#     # ─────────────────────────────────────────────────────────────

#     async def _update_agent_metrics(self) -> Dict[str, Any]:
#         """Update agent performance metrics and adapt LR."""
#         try:
#             if self.episode_rewards:
#                 avg_reward = float(np.mean(list(self.episode_rewards)[-10:]))
#                 performance_score = max(
#                     0.0, min(1.0, (avg_reward + 100.0) / 200.0)
#                 )
#             else:
#                 avg_reward = 0.0
#                 performance_score = 0.0

#             self.training_stats["avg_episode_reward"] = avg_reward

#             # Learning rate adaptation
#             if self.episode_rewards:
#                 self.lr_scheduler.step(avg_reward)
#                 current_lr = float(self.optimizer.param_groups[0]["lr"])
#                 self.training_stats["learning_rate"] = current_lr
#             else:
#                 current_lr = self.training_stats.get("learning_rate", self._cfg.learning_rate)

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] _update_agent_metrics: "
#                     f"avg_reward={avg_reward:.2f}, performance_score={performance_score:.3f}, "
#                     f"learning_rate={current_lr:.2e}"
#                 )

#             return {
#                 "agent_metrics": {
#                     "performance_score": performance_score,
#                     "average_reward": avg_reward,
#                     "episodes_completed": self.training_stats["episodes_completed"],
#                     "training_updates": self.training_stats["total_updates"],
#                     "exploration_level": self.action_statistics["exploration_level"],
#                     "learning_rate": current_lr,
#                 }
#             }

#         except Exception as e:
#             self.logger.error(f"Agent metrics update failed: {e}")
#             return {"agent_metrics": {"error": str(e)}}

#     async def _generate_ppo_thesis(
#         self, ppo_data: Dict[str, Any], result: Dict[str, Any]
#     ) -> str:
#         """Generate a human-readable thesis summarizing PPO status."""
#         try:
#             episodes = self.training_stats["episodes_completed"]
#             updates = self.training_stats["total_updates"]
#             avg_reward = self.training_stats["avg_episode_reward"]

#             policy_loss = self.training_stats["policy_loss_trend"]
#             value_loss = self.training_stats["value_loss_trend"]
#             explained_var = self.training_stats["explained_variance"]

#             thesis_parts = [
#                 f"PPO Agent Performance: {episodes} episodes completed with {updates} policy updates",
#                 f"Average reward: {avg_reward:.2f} with explained variance {explained_var:.2f}",
#                 f"Learning progress: Policy loss {policy_loss:.4f}, Value loss {value_loss:.4f}",
#             ]

#             if result.get("action_selected", False):
#                 action = result.get("action", [0.0, 0.0])
#                 value_est = result.get("value_estimate", 0.0)
#                 thesis_parts.append(
#                     f"Action selected: [{action[0]:.3f}, {action[1]:.3f}] "
#                     f"with value estimate {value_est:.3f}"
#                 )

#             if result.get("training_performed", False):
#                 update_result = result.get("update_result", {})
#                 grad_norm = update_result.get("gradient_norm", 0.0)
#                 thesis_parts.append(
#                     f"Policy updated with gradient norm {grad_norm:.4f}"
#                 )

#             exploration = self.action_statistics["exploration_level"]
#             thesis_parts.append(
#                 f"Exploration level: {exploration:.2f} maintaining learning diversity"
#             )

#             current_lr = self.training_stats["learning_rate"]
#             if current_lr != self._cfg.learning_rate:
#                 thesis_parts.append(
#                     f"Learning rate adapted to {current_lr:.2e} for optimization"
#                 )

#             if len(self.episode_rewards) > 10:
#                 recent_trend = np.mean(
#                     list(self.episode_rewards)[-5:]
#                 ) - np.mean(list(self.episode_rewards)[-10:-5])
#                 if recent_trend > 0.1:
#                     thesis_parts.append("Recent performance trend: IMPROVING")
#                 elif recent_trend < -0.1:
#                     thesis_parts.append("Recent performance trend: DECLINING")
#                 else:
#                     thesis_parts.append("Recent performance trend: STABLE")

#             return " | ".join(thesis_parts)

#         except Exception as e:
#             return (
#                 f"PPO thesis generation failed: {str(e)} - "
#                 f"Agent continuing with basic functionality"
#             )

#     # ─────────────────────────────────────────────────────────────
#     # SmartInfoBus publishing
#     # ─────────────────────────────────────────────────────────────

#     async def _update_ppo_smart_bus(self, result: Dict[str, Any], thesis: str):
#         """Update SmartInfoBus with PPO results and arbiter decision."""
#         try:
#             # Policy actions
#             if result.get("action_selected", False):
#                 action_data = {
#                     "action": result.get("action", [0.0, 0.0]),
#                     "log_prob": result.get("log_prob", 0.0),
#                     "value_estimate": result.get("value_estimate", 0.0),
#                     "action_std": result.get(
#                         "action_std",
#                         self._last_action_std or [1.0] * int(self._cfg.act_size),
#                     ),
#                     "exploration_level": self.action_statistics["exploration_level"],
#                 }

#                 self.smart_bus.set(
#                     "policy_actions",
#                     action_data,
#                     module="PPOAgent",
#                     thesis=thesis,
#                 )

#                 # Back-compat simple actions vector
#                 try:
#                     self.smart_bus.set(
#                         "actions",
#                         result.get("action", [0.0] * int(self._cfg.act_size)),
#                         module="PPOAgent",
#                         thesis="Raw PPO action vector for compatibility",
#                     )
#                 except Exception:
#                     pass

#             # Agent performance
#             agent_metrics = result.get("agent_metrics", {})
#             performance_data = {
#                 "performance_score": agent_metrics.get("performance_score", 0.0),
#                 "average_reward": agent_metrics.get("average_reward", 0.0),
#                 "episodes_completed": agent_metrics.get(
#                     "episodes_completed", 0
#                 ),
#                 "training_updates": agent_metrics.get(
#                     "training_updates", 0
#                 ),
#                 "learning_rate": agent_metrics.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#             }

#             self.smart_bus.set(
#                 "agent_performance",
#                 performance_data,
#                 module="PPOAgent",
#                 thesis="PPO agent performance metrics and learning progress",
#             )

#             # Training metrics
#             if result.get("training_performed", False):
#                 training_data = {
#                     "policy_loss": self.training_stats["policy_loss_trend"],
#                     "value_loss": self.training_stats["value_loss_trend"],
#                     "entropy": self.training_stats["entropy_trend"],
#                     "gradient_norm": self.training_stats["gradient_norm"],
#                     "explained_variance": self.training_stats["explained_variance"],
#                     "total_updates": self.training_stats["total_updates"],
#                 }

#                 self.smart_bus.set(
#                     "training_metrics",
#                     training_data,
#                     module="PPOAgent",
#                     thesis="PPO training metrics and optimization progress",
#                 )

#             # Policy gradients info
#             try:
#                 network_params = sum(p.numel() for p in self.network.parameters())
#             except Exception:
#                 network_params = 0

#             gradient_data = {
#                 "gradient_norm": self.training_stats["gradient_norm"],
#                 "learning_rate": self.training_stats["learning_rate"],
#                 "network_parameters": network_params,
#                 "forward_passes": self._neural_performance["forward_passes"],
#                 "backward_passes": self._neural_performance["backward_passes"],
#             }

#             self.smart_bus.set(
#                 "policy_gradients",
#                 gradient_data,
#                 module="PPOAgent",
#                 thesis="Policy gradient information and neural network performance",
#             )

#             # Publish observations (last snapshot) if available
#             if isinstance(self._last_obs_vec, np.ndarray):
#                 try:
#                     self.smart_bus.set(
#                         "observations",
#                         self._last_obs_vec.astype(float).tolist(),
#                         module="PPOAgent",
#                         thesis="Latest observation vector snapshot",
#                     )
#                 except Exception:
#                     pass

#             # Publish recent rewards snapshot
#             if self._recent_rewards:
#                 try:
#                     self.smart_bus.set(
#                         "rewards",
#                         list(self._recent_rewards)[-10:],
#                         module="PPOAgent",
#                         thesis="Recent rewards window for compatibility",
#                     )
#                 except Exception:
#                     pass

#             # Training signals / data compatibility payloads
#             compat_training_signals = {
#                 "gradient_norm": self.training_stats["gradient_norm"],
#                 "explained_variance": self.training_stats["explained_variance"],
#                 "policy_loss": self.training_stats["policy_loss_trend"],
#                 "value_loss": self.training_stats["value_loss_trend"],
#             }
#             try:
#                 self.smart_bus.set(
#                     "training_signals",
#                     compat_training_signals,
#                     module="PPOAgent",
#                     thesis="Training signals for downstream consumers",
#                 )
#             except Exception:
#                 pass

#             try:
#                 self.smart_bus.set(
#                     "training_data",
#                     {
#                         "buffer_sizes": {
#                             k: len(v) if hasattr(v, "__len__") else 0
#                             for k, v in self.buffer.items()
#                         },
#                         "total_updates": self.training_stats["total_updates"],
#                     },
#                     module="PPOAgent",
#                     thesis="PPO buffer snapshot and update counters",
#                 )
#             except Exception:
#                 pass

#             # Voting additions: publish namespaced proposal + confidence
#             if "PPOAgent_voting_proposal" in result:
#                 try:
#                     self.smart_bus.set(
#                         "PPOAgent_voting_proposal",
#                         result["PPOAgent_voting_proposal"],
#                         module="PPOAgent",
#                         thesis="PPOAgent voting payload (direction/magnitude/confidence)",
#                     )
#                 except Exception:
#                     pass

#             if "PPOAgent_confidence" in result:
#                 try:
#                     self.smart_bus.set(
#                         "PPOAgent_confidence",
#                         float(result["PPOAgent_confidence"]),
#                         module="PPOAgent",
#                         thesis="PPOAgent voting confidence",
#                     )
#                 except Exception:
#                     pass

#             # ═══════════════════════════════════════════════════════════════════
#             # INTELLIGENT ARBITER: Publish final decision keys
#             # ═══════════════════════════════════════════════════════════════════
#             if "ppo_final_decision" in result:
#                 final_dec = result["ppo_final_decision"] or {}

#                 # Canonical arbiter payload
#                 try:
#                     self.smart_bus.set(
#                         "ppo_final_decision",
#                         final_dec,
#                         module="PPOAgent",
#                         thesis=final_dec.get(
#                             "reasoning", "PPOAgent final GO/NO-GO decision"
#                         ),
#                     )
#                 except Exception:
#                     pass

#                 gate_val = bool(final_dec.get("gate_passed", False))
#                 direction_val = final_dec.get("direction", "hold")
#                 pos_size_val = float(
#                     final_dec.get(
#                         "position_size", result.get("ppo_position_size", 0.0)
#                     )
#                 )

#                 # Namespaced gate + size (for PositionManager)
#                 try:
#                     self.smart_bus.set(
#                         "ppo_gate_passed",
#                         gate_val,
#                         module="PPOAgent",
#                         thesis="Whether PPOAgent approved the trade",
#                     )
#                 except Exception:
#                     pass

#                 try:
#                     self.smart_bus.set(
#                         "ppo_position_size",
#                         pos_size_val,
#                         module="PPOAgent",
#                         thesis=f"PPOAgent position size: {pos_size_val:.4f}",
#                     )
#                 except Exception:
#                     pass

#                 # Global aliases for other consumers
#                 try:
#                     self.smart_bus.set(
#                         "gate_passed",
#                         gate_val,
#                         module="PPOAgent",
#                         thesis="Global arbiter gate (alias)",
#                     )
#                 except Exception:
#                     pass

#                 try:
#                     self.smart_bus.set(
#                         "trading_signal",
#                         direction_val,
#                         module="PPOAgent",
#                         thesis=f"PPOAgent trading direction: {direction_val}",
#                     )
#                 except Exception:
#                     pass

#                 # Combined arbiter decision object
#                 try:
#                     self.smart_bus.set(
#                         "arbiter_decision",
#                         {
#                             "approved": gate_val,
#                             "direction": direction_val,
#                             "confidence": float(
#                                 final_dec.get("confidence", 0.0)
#                             ),
#                             "position_size": pos_size_val,
#                             "reasoning": final_dec.get("reasoning", ""),
#                             "source": "PPOAgent",
#                         },
#                         module="PPOAgent",
#                         thesis="Arbiter decision payload for downstream modules",
#                     )
#                 except Exception:
#                     pass

#         except Exception as e:
#             self.logger.error(f"Failed to update SmartInfoBus: {e}")

#     # ─────────────────────────────────────────────────────────────
#     # Fallbacks / error handling
#     # ─────────────────────────────────────────────────────────────

#     async def _handle_no_data_fallback(self) -> Dict[str, Any]:
#         """Handle case when no PPO data is available."""
#         self.logger.warning("No PPO data available - returning current state")
#         thesis = "PPO Agent: no input data, returning last known state and defaults"

#         act_size = int(getattr(self._cfg, "act_size", 2) or 2)
#         obs_size = int(getattr(self._cfg, "obs_size", 10) or 10)

#         last_action = (
#             self.last_action.tolist()
#             if hasattr(self.last_action, "tolist")
#             else [0.0] * act_size
#         )

#         observations = None
#         try:
#             if isinstance(self._last_obs_vec, np.ndarray):
#                 observations = self._last_obs_vec.astype(float).tolist()
#             elif isinstance(self._last_obs_vec, list):
#                 observations = self._last_obs_vec
#         except Exception:
#             observations = None

#         fallback_vote = {
#             "member": "PPOAgent",
#             "action": "flat",
#             "proposal": {
#                 "direction": "flat",
#                 "magnitude": 0.0,
#                 "horizon": "intraday",
#             },
#             "confidence": 0.2,
#             "rationale": thesis,
#             "timestamp": datetime.now().isoformat(),
#         }

#         final_decision = {
#             "direction": "hold",
#             "confidence": 0.0,
#             "reasoning": "No PPO data available, arbiter in safe HOLD mode",
#             "trust_score": 0.0,
#             "committee_action": "hold",
#             "committee_confidence": 0.0,
#             "expert_consensus": "flat",
#             "expert_confidence": 0.0,
#             "regime": "unknown",
#             "value_estimate": 0.0,
#             "gate_passed": False,
#             "position_size": 0.0,
#         }

#         return {
#             "policy_actions": {
#                 "action": last_action,
#                 "log_prob": 0.0,
#                 "value_estimate": 0.0,
#                 "action_std": self._last_action_std or [1.0] * act_size,
#                 "exploration_level": self.action_statistics.get(
#                     "exploration_level", 0.5
#                 ),
#             },
#             "agent_performance": {
#                 "performance_score": 0.0,
#                 "average_reward": self.training_stats.get(
#                     "avg_episode_reward", 0.0
#                 ),
#                 "episodes_completed": self.training_stats.get(
#                     "episodes_completed", 0
#                 ),
#                 "training_updates": self.training_stats.get("total_updates", 0),
#                 "learning_rate": self.training_stats.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#             },
#             "training_metrics": {
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#                 "entropy": self.training_stats.get("entropy_trend", 0.0),
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             },
#             "policy_gradients": {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "learning_rate": self.training_stats.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#                 "network_parameters": 0,
#                 "forward_passes": self._neural_performance.get("forward_passes", 0),
#                 "backward_passes": self._neural_performance.get("backward_passes", 0),
#             },
#             "actions": last_action,
#             "training_data": {
#                 "buffer_sizes": {
#                     k: (len(v) if hasattr(v, "__len__") else 0)
#                     for k, v in self.buffer.items()
#                 },
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             },
#             "observations": observations or [0.0] * obs_size,
#             "rewards": list(self._recent_rewards)[-10:]
#             if self._recent_rewards
#             else [],
#             "training_signals": {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#             },
#             "PPOAgent_voting_proposal": fallback_vote,
#             "PPOAgent_confidence": 0.2,
#             "ppo_final_decision": final_decision,
#             "ppo_gate_passed": False,
#             "ppo_position_size": 0.0,
#             "_thesis": thesis,
#             "fallback_reason": "no_ppo_data",
#         }

#     async def _handle_ppo_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
#         """Handle PPO errors and return a safe fallback response."""
#         processing_time = (time.time() - start_time) * 1000.0

#         # Update circuit breaker
#         self.circuit_breaker["failures"] += 1
#         self.circuit_breaker["last_failure"] = time.time()

#         if self.circuit_breaker["failures"] >= self.circuit_breaker["threshold"]:
#             self.circuit_breaker["state"] = "OPEN"

#         error_context = self.error_pinpointer.analyze_error(error, "PPOAgent")
#         explanation = self.english_explainer.explain_error(
#             "PPOAgent", str(error), "PPO training"
#         )

#         self.logger.error(
#             format_operator_message(
#                 "[CRASH]",
#                 "PPO_AGENT_ERROR",
#                 error=str(error),
#                 details=explanation,
#                 processing_time_ms=processing_time,
#                 context="ppo_training",
#             )
#         )

#         self._record_failure(error)

#         return self._create_fallback_response(f"error: {str(error)}")

#     def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
#         """Create fallback response for error cases."""
#         act_size = int(getattr(self._cfg, "act_size", 2) or 2)
#         obs_size = int(getattr(self._cfg, "obs_size", 10) or 10)

#         last_action = (
#             self.last_action.tolist()
#             if hasattr(self.last_action, "tolist")
#             else [0.0] * act_size
#         )
#         thesis = self.english_explainer.explain_error(
#             "PPOAgent", f"Fallback due to {reason}", "PPO processing"
#         )

#         observations = None
#         try:
#             if isinstance(self._last_obs_vec, np.ndarray):
#                 observations = self._last_obs_vec.astype(float).tolist()
#             elif isinstance(self._last_obs_vec, list):
#                 observations = self._last_obs_vec
#         except Exception:
#             observations = None

#         try:
#             network_params = sum(p.numel() for p in self.network.parameters())
#         except Exception:
#             network_params = 0

#         vote_payload = {
#             "member": "PPOAgent",
#             "action": "flat",
#             "proposal": {
#                 "direction": "flat",
#                 "magnitude": 0.0,
#                 "horizon": "intraday",
#             },
#             "confidence": 0.1,
#             "rationale": thesis,
#             "timestamp": datetime.now().isoformat(),
#             "meta": {"circuit_breaker": self.circuit_breaker["state"]},
#         }

#         final_decision = {
#             "direction": "hold",
#             "confidence": 0.0,
#             "reasoning": f"Error in arbiter/ppo: {reason}",
#             "trust_score": 0.0,
#             "committee_action": "hold",
#             "committee_confidence": 0.0,
#             "expert_consensus": "flat",
#             "expert_confidence": 0.0,
#             "regime": "unknown",
#             "value_estimate": 0.0,
#             "gate_passed": False,
#             "position_size": 0.0,
#         }

#         return {
#             "policy_actions": {
#                 "action": last_action,
#                 "log_prob": 0.0,
#                 "value_estimate": 0.0,
#                 "action_std": self._last_action_std or [1.0] * act_size,
#                 "exploration_level": self.action_statistics.get(
#                     "exploration_level", 0.5
#                 ),
#             },
#             "agent_performance": {
#                 "performance_score": 0.0,
#                 "average_reward": self.training_stats.get(
#                     "avg_episode_reward", 0.0
#                 ),
#                 "episodes_completed": self.training_stats.get(
#                     "episodes_completed", 0
#                 ),
#                 "training_updates": self.training_stats.get("total_updates", 0),
#                 "learning_rate": self.training_stats.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#             },
#             "training_metrics": {
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#                 "entropy": self.training_stats.get("entropy_trend", 0.0),
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             },
#             "policy_gradients": {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "learning_rate": self.training_stats.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#                 "network_parameters": network_params,
#                 "forward_passes": self._neural_performance.get(
#                     "forward_passes", 0
#                 ),
#                 "backward_passes": self._neural_performance.get(
#                     "backward_passes", 0
#                 ),
#             },
#             "actions": last_action,
#             "training_data": {
#                 "buffer_sizes": {
#                     k: (len(v) if hasattr(v, "__len__") else 0)
#                     for k, v in self.buffer.items()
#                 },
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             },
#             "observations": observations or [0.0] * obs_size,
#             "rewards": list(self._recent_rewards)[-10:]
#             if self._recent_rewards
#             else [],
#             "training_signals": {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#             },
#             "PPOAgent_voting_proposal": vote_payload,
#             "PPOAgent_confidence": 0.1,
#             "ppo_final_decision": final_decision,
#             "ppo_gate_passed": False,
#             "ppo_position_size": 0.0,
#             "_thesis": thesis,
#             "circuit_breaker_state": self.circuit_breaker["state"],
#             "fallback_reason": reason,
#         }

#     # ─────────────────────────────────────────────────────────────
#     # Health / monitoring
#     # ─────────────────────────────────────────────────────────────

#     def _update_ppo_health(self):
#         """Update PPO health metrics."""
#         try:
#             if not hasattr(self, "episode_rewards") or not hasattr(
#                 self, "training_stats"
#             ):
#                 return

#             if len(self.episode_rewards) > 20:
#                 recent_performance = float(
#                     np.mean(list(self.episode_rewards)[-10:])
#                 )
#                 if recent_performance < -50:
#                     self._health_status = "warning"
#                 elif recent_performance > 50:
#                     self._health_status = "healthy"

#             if self.training_stats["gradient_norm"] > 10.0:
#                 self._health_status = "warning"

#             self._last_health_check = time.time()

#         except Exception as e:
#             self.logger.error(f"Health check failed: {e}")
#             self._health_status = "warning"

#     def _analyze_learning_progress(self):
#         """Analyze learning progress trends."""
#         try:
#             if not hasattr(self, "episode_rewards"):
#                 return

#             if len(self.episode_rewards) >= 20:
#                 recent_rewards = list(self.episode_rewards)[-10:]
#                 older_rewards = list(self.episode_rewards)[-20:-10]

#                 recent_avg = float(np.mean(recent_rewards))
#                 older_avg = float(np.mean(older_rewards))

#                 improvement = recent_avg - older_avg

#                 if improvement > 5.0:
#                     self.logger.info(
#                         format_operator_message(
#                             "[CHART]",
#                             "LEARNING_PROGRESS_GOOD",
#                             improvement=f"{improvement:.2f}",
#                             recent_avg=f"{recent_avg:.2f}",
#                             context="learning_analysis",
#                         )
#                     )
#                 elif improvement < -5.0:
#                     self.logger.warning(
#                         format_operator_message(
#                             "📉",
#                             "LEARNING_REGRESSION",
#                             regression=f"{improvement:.2f}",
#                             recent_avg=f"{recent_avg:.2f}",
#                             context="learning_analysis",
#                         )
#                     )

#         except Exception as e:
#             self.logger.error(f"Learning progress analysis failed: {e}")

#     def _record_success(self, processing_time: float):
#         """Record successful processing."""
#         self.performance_tracker.record_metric(
#             "PPOAgent", "processing_cycle", processing_time, True
#         )

#         if self.circuit_breaker["state"] == "OPEN":
#             self.circuit_breaker["failures"] = 0
#             self.circuit_breaker["state"] = "CLOSED"

#     def _record_failure(self, error: Exception):
#         """Record processing failure."""
#         self.performance_tracker.record_metric(
#             "PPOAgent", "processing_cycle", 0.0, False
#         )

#     # ─────────────────────────────────────────────────────────────
#     # REQUIRED OUTPUT NORMALIZATION
#     # ─────────────────────────────────────────────────────────────

#     def _ensure_required_outputs(self, result: Dict[str, Any]) -> None:
#         """Ensure all required output keys exist in result dict."""
#         act_size = int(getattr(self._cfg, "act_size", 2) or 2)
#         obs_size = int(getattr(self._cfg, "obs_size", 10) or 10)

#         # policy_actions
#         if "policy_actions" not in result:
#             action_vec_out = result.get("action") or (
#                 self.last_action.tolist()
#                 if hasattr(self.last_action, "tolist")
#                 else [0.0] * act_size
#             )
#             result["policy_actions"] = {
#                 "action": action_vec_out,
#                 "log_prob": result.get("log_prob", 0.0),
#                 "value_estimate": result.get("value_estimate", 0.0),
#                 "action_std": result.get(
#                     "action_std", self._last_action_std or [1.0] * act_size
#                 ),
#                 "exploration_level": self.action_statistics.get(
#                     "exploration_level", 0.5
#                 ),
#             }

#         # agent_performance
#         if "agent_performance" not in result:
#             agent_metrics = result.get("agent_metrics", {})
#             result["agent_performance"] = {
#                 "performance_score": agent_metrics.get("performance_score", 0.0),
#                 "average_reward": agent_metrics.get(
#                     "average_reward", self.training_stats.get("avg_episode_reward", 0.0)
#                 ),
#                 "episodes_completed": agent_metrics.get(
#                     "episodes_completed", self.training_stats.get("episodes_completed", 0)
#                 ),
#                 "training_updates": agent_metrics.get(
#                     "training_updates", self.training_stats.get("total_updates", 0)
#                 ),
#                 "learning_rate": agent_metrics.get(
#                     "learning_rate",
#                     self.training_stats.get(
#                         "learning_rate", self._cfg.learning_rate
#                     ),
#                 ),
#             }

#         # training_metrics
#         if "training_metrics" not in result:
#             result["training_metrics"] = {
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#                 "entropy": self.training_stats.get("entropy_trend", 0.0),
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             }

#         # policy_gradients
#         if "policy_gradients" not in result:
#             try:
#                 network_params = sum(p.numel() for p in self.network.parameters())
#             except Exception:
#                 network_params = 0
#             result["policy_gradients"] = {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "learning_rate": self.training_stats.get(
#                     "learning_rate", self._cfg.learning_rate
#                 ),
#                 "network_parameters": network_params,
#                 "forward_passes": self._neural_performance.get("forward_passes", 0),
#                 "backward_passes": self._neural_performance.get("backward_passes", 0),
#             }

#         # actions
#         if "actions" not in result:
#             result["actions"] = result.get("action") or (
#                 self.last_action.tolist()
#                 if hasattr(self.last_action, "tolist")
#                 else [0.0] * act_size
#             )

#         # training_data
#         if "training_data" not in result:
#             try:
#                 buffer_sizes = {
#                     k: (len(v) if hasattr(v, "__len__") else 0)
#                     for k, v in self.buffer.items()
#                 }
#             except Exception:
#                 buffer_sizes = {}
#             result["training_data"] = {
#                 "buffer_sizes": buffer_sizes,
#                 "total_updates": self.training_stats.get("total_updates", 0),
#             }

#         # observations
#         if "observations" not in result:
#             obs_vec = None
#             try:
#                 if isinstance(self._last_obs_vec, np.ndarray):
#                     obs_vec = self._last_obs_vec.astype(float).tolist()
#                 elif isinstance(self._last_obs_vec, list):
#                     obs_vec = self._last_obs_vec
#             except Exception:
#                 obs_vec = None
#             result["observations"] = obs_vec or [0.0] * obs_size

#         # rewards
#         if "rewards" not in result:
#             result["rewards"] = (
#                 list(self._recent_rewards)[-10:] if self._recent_rewards else []
#             )

#         # training_signals
#         if "training_signals" not in result:
#             result["training_signals"] = {
#                 "gradient_norm": self.training_stats.get("gradient_norm", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "policy_loss": self.training_stats.get("policy_loss_trend", 0.0),
#                 "value_loss": self.training_stats.get("value_loss_trend", 0.0),
#             }

#     # ─────────────────────────────────────────────────────────────
#     # COMMITTEE / EXPERT SIGNALS
#     # ─────────────────────────────────────────────────────────────

#     def _gather_expert_signals(self) -> Dict[str, Any]:
#         """
#         Gather all expert voting signals from the bus.

#         Returns a dict with expert proposals, confidences, risk signals, and memory.
#         """
#         name = "PPOAgent"

#         def _expert_block(vote_key: str, conf_key: str) -> Dict[str, Any]:
#             raw = self.smart_bus.get(vote_key, name) or "flat"
#             conf = self.smart_bus.get(conf_key, name)
#             try:
#                 conf_val = float(conf) if conf is not None else 0.0
#             except Exception:
#                 conf_val = 0.0
#             return {
#                 "proposal": raw,
#                 "confidence": conf_val,
#             }

#         expert_signals = {
#             "trend": _expert_block(
#                 "TrendExpert_voting_proposal", "TrendExpert_confidence"
#             ),
#             "momentum": _expert_block(
#                 "MomentumExpert_voting_proposal", "MomentumExpert_confidence"
#             ),
#             "theme": _expert_block(
#                 "ThemeExpert_voting_proposal", "ThemeExpert_confidence"
#             ),
#             "seasonality": _expert_block(
#                 "SeasonalityRiskExpert_voting_proposal",
#                 "SeasonalityRiskExpert_confidence",
#             ),
#         }

#         market_context = {
#             "regime": self.smart_bus.get("market_regime", name) or "unknown",
#             "regime_strength": float(
#                 self.smart_bus.get("regime_strength", name) or 0.5
#             ),
#         }

#         risk_signals = {
#             "risk_data": self.smart_bus.get("risk_data", name) or {},
#             "portfolio_risk": self.smart_bus.get("portfolio_risk", name) or {},
#         }

#         raw_memory_gate = self.smart_bus.get("memory_gate", name)
#         raw_danger_zones = self.smart_bus.get("danger_zones", name)

#         if isinstance(raw_memory_gate, dict):
#             try:
#                 memory_gate_value = float(raw_memory_gate.get("risk_multiplier", 1.0))
#             except Exception:
#                 memory_gate_value = 1.0
#             memory_gate_meta = raw_memory_gate
#         else:
#             try:
#                 memory_gate_value = (
#                     float(raw_memory_gate) if raw_memory_gate is not None else 1.0
#                 )
#             except Exception:
#                 memory_gate_value = 1.0
#             memory_gate_meta = {
#                 "risk_multiplier": memory_gate_value,
#                 "veto": False,
#                 "reasons": [],
#             }

#         if isinstance(raw_danger_zones, dict):
#             dz_dict = raw_danger_zones
#         elif isinstance(raw_danger_zones, list):
#             dz_dict = {
#                 "zones": raw_danger_zones,
#                 "zone_count": len(raw_danger_zones),
#             }
#         else:
#             dz_dict = {
#                 "zones": [],
#                 "zone_count": 0,
#             }

#         try:
#             dz_count = int(dz_dict.get("zone_count", 0))
#         except Exception:
#             zones = dz_dict.get("zones", [])
#             dz_count = len(zones) if isinstance(zones, list) else 0
#         dz_dict["zone_count"] = dz_count

#         memory_signals = {
#             "memory_gate": memory_gate_meta,
#             "memory_gate_value": memory_gate_value,
#             "danger_zones": dz_dict,
#         }

#         if self.debug:
#             self.logger.debug(
#                 f"[DEBUG][PPO] _gather_expert_signals: regime={market_context['regime']}, "
#                 f"regime_strength={market_context['regime_strength']:.2f}, "
#                 f"memory_gate={memory_gate_value:.2f}, danger_zone_count={dz_count}"
#             )

#         return {
#             "experts": expert_signals,
#             "market": market_context,
#             "risk": risk_signals,
#             "memory": memory_signals,
#         }

#     def _compute_expert_consensus(
#         self, expert_signals: Dict[str, Any]
#     ) -> Tuple[str, float]:
#         """
#         Compute consensus direction and confidence from expert signals.

#         Returns (consensus_direction, consensus_confidence).
#         """
#         experts = expert_signals.get("experts", {})

#         long_score = 0.0
#         short_score = 0.0
#         total_weight = 0.0

#         for expert_name, sig in experts.items():
#             raw_prop = sig.get("proposal", "flat")
#             conf = float(sig.get("confidence", 0.0))

#             if isinstance(raw_prop, dict):
#                 direction = (
#                     raw_prop.get("direction")
#                     or raw_prop.get("action")
#                     or raw_prop.get("global_direction")
#                 )
#             elif isinstance(raw_prop, str):
#                 direction = raw_prop
#             else:
#                 direction = str(raw_prop)

#             proposal = str(direction or "flat").lower()

#             if proposal in ("long", "buy", "bullish"):
#                 long_score += conf
#             elif proposal in ("short", "sell", "bearish"):
#                 short_score += conf

#             total_weight += max(conf, 0.0)

#         if total_weight < 1e-6:
#             return "flat", 0.0

#         if long_score > short_score + 0.2:
#             direction = "long"
#             consensus_conf = long_score / total_weight
#         elif short_score > long_score + 0.2:
#             direction = "short"
#             consensus_conf = short_score / total_weight
#         else:
#             direction = "flat"
#             consensus_conf = 0.3

#         consensus_conf = float(np.clip(consensus_conf, 0.0, 1.0))

#         if self.debug:
#             self.logger.debug(
#                 f"[DEBUG][PPO] _compute_expert_consensus: direction={direction}, "
#                 f"confidence={consensus_conf:.2f}"
#             )

#         return direction, consensus_conf

#     def _gather_committee_consensus(self) -> Dict[str, Any]:
#         """
#         Gather committee consensus from the bus.
#         Committee has already aggregated expert votes.
#         """
#         name = "PPOAgent"

#         committee_decision = self.smart_bus.get("committee_decision", name) or {}
#         if isinstance(committee_decision, str):
#             committee_decision = {"action": committee_decision}

#         data = {
#             "action": str(committee_decision.get("action", "hold")).lower(),
#             "confidence": float(
#                 self.smart_bus.get("committee_confidence", name) or 0.5
#             ),
#             "consensus_score": float(
#                 self.smart_bus.get("consensus_score", name) or 0.5
#             ),
#             "fragility": float(self.smart_bus.get("fragility", name) or 0.5),
#         }

#         if self.debug:
#             self.logger.debug(
#                 f"[DEBUG][PPO] _gather_committee_consensus: {data}"
#             )

#         return data

#     # ─────────────────────────────────────────────────────────────
#     # INTELLIGENT ARBITER: Final GO/NO-GO decision
#     # ─────────────────────────────────────────────────────────────

#     async def make_final_decision(
#         self, observation: Any = None, **inputs
#     ) -> Dict[str, Any]:
#         """
#         INTELLIGENT ARBITER: Make the final GO/NO-GO trading decision.

#         The core logic:
#         1. Gather committee consensus and expert signals.
#         2. Build a unified observation (48-dim) using PPOObservationBuilder
#            or use the provided observation if already built.
#         3. Run PPO policy to produce:
#              - trust_score (action[0])
#              - position_size_score (action[1])
#         4. Interpret these into:
#              - final direction (follow/override/uncertain)
#              - confidence
#              - gate_passed (GO/NO-GO)
#              - position_size (0..1)
#         5. Apply soft risk/memory adjustments (no double-veto).
#         """
#         try:
#             # 1) Gather inputs
#             committee = self._gather_committee_consensus()
#             expert_signals = self._gather_expert_signals()
#             expert_consensus, expert_confidence = self._compute_expert_consensus(
#                 expert_signals
#             )

#             memory_gate_val = expert_signals["memory"].get("memory_gate", 1.0)
#             if isinstance(memory_gate_val, dict):
#                 memory_gate_val = float(
#                     memory_gate_val.get(
#                         "risk_multiplier", memory_gate_val.get("value", 1.0)
#                     )
#                 )
#             memory_gate_val = float(memory_gate_val)

#             danger_zones = expert_signals["memory"].get("danger_zones", {})
#             regime = expert_signals["market"].get("regime", "unknown")
#             fragility = committee.get("fragility", 0.5)

#             # Diagnostic arbiter view (for logs / debugging only)
#             arbiter_view = self._build_arbiter_observation(
#                 committee=committee,
#                 expert_signals=expert_signals,
#                 expert_consensus=expert_consensus,
#                 expert_confidence=expert_confidence,
#                 memory_gate=memory_gate_val,
#                 fragility=fragility,
#             )

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO ARBITER] arbiter_view={arbiter_view}"
#                 )

#             # 2) Build observation for PPO policy
#             try:
#                 if observation is not None:
#                     obs_vec = self._normalize_observation(observation)
#                     obs_source = "external_observation"
#                 else:
#                     obs_vec = self.build_unified_observation()
#                     obs_source = "unified_builder"

#                 self._last_obs_vec = obs_vec.copy()

#                 if self.debug:
#                     self.logger.debug(
#                         f"[DEBUG][PPO ARBITER] using obs_source={obs_source}, "
#                         f"shape={obs_vec.shape}, "
#                         f"min={float(np.min(obs_vec)):.4f}, max={float(np.max(obs_vec)):.4f}"
#                     )
#             except Exception as e:
#                 self.logger.warning(
#                     f"[PPO ARBITER] Observation build failed ({e}), using zeros"
#                 )
#                 obs_vec = np.zeros(self._cfg.obs_size, dtype=np.float32)

#             # 3) PPO policy output
#             with torch.no_grad():
#                 obs_tensor = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)
#                 action_mean, action_log_std, value_t = self.network(obs_tensor)
#                 action_vec = action_mean.squeeze(0).cpu().numpy()

#             trust_score = float(action_vec[0]) if len(action_vec) > 0 else 0.0
#             size_score = float(action_vec[1]) if len(action_vec) > 1 else 0.0
#             value_estimate = float(value_t.squeeze().cpu().item())

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO ARBITER] raw_action_vec={action_vec.tolist()}, "
#                     f"trust_score={trust_score:.3f}, size_score={size_score:.3f}, "
#                     f"value_estimate={value_estimate:.4f}"
#                 )

#             # 4) Interpret policy output into decision
#             committee_action = committee.get("action", "hold")
#             committee_confidence = committee.get("confidence", 0.5)

#             if trust_score > 0.3:
#                 final_action = committee_action
#                 final_confidence = committee_confidence * (0.7 + 0.3 * trust_score)
#                 reasoning = f"PPO trusts committee ({trust_score:.2f}): {committee_action}"
#                 gate_passed = committee_action in ("long", "short", "buy", "sell")
#             elif trust_score < -0.3:
#                 final_action = "hold"
#                 final_confidence = 0.3
#                 reasoning = f"PPO overrides committee ({trust_score:.2f}): going flat"
#                 gate_passed = False
#             else:
#                 final_action = committee_action
#                 final_confidence = committee_confidence * 0.5
#                 reasoning = (
#                     f"PPO uncertain ({trust_score:.2f}): "
#                     f"following committee with reduced confidence"
#                 )
#                 gate_passed = (
#                     committee_action in ("long", "short", "buy", "sell")
#                     and final_confidence > 0.4
#                 )

#             # 5) Apply risk/memory gates (soft adjustments only)
#             if memory_gate_val < 0.7:
#                 final_confidence *= memory_gate_val
#                 reasoning += f" | MEMORY_CAUTION (risk_mult={memory_gate_val:.2f})"

#             danger_zone_count = 0
#             if isinstance(danger_zones, dict):
#                 try:
#                     danger_zone_count = int(danger_zones.get("zone_count", 0))
#                 except Exception:
#                     zones = danger_zones.get("zones", [])
#                     danger_zone_count = len(zones) if isinstance(zones, list) else 0
#             elif isinstance(danger_zones, list):
#                 danger_zone_count = len(danger_zones)

#             if danger_zone_count > 0:
#                 final_confidence *= 0.7
#                 reasoning += f" | DANGER_ZONE_NEARBY (n={danger_zone_count})"

#             if fragility > 0.90:
#                 reasoning += f" | (fragility={fragility:.2f})"

#             # 6) Calculate position size
#             raw_size = (size_score + 1.0) / 2.0  # map [-1,1] -> [0,1]
#             final_confidence = max(0.0, final_confidence)
#             position_size = float(
#                 np.clip(raw_size * final_confidence, 0.0, 1.0)
#             )
#             if not gate_passed:
#                 position_size = 0.0

#             final_decision = {
#                 "direction": final_action,
#                 "confidence": float(final_confidence),
#                 "reasoning": reasoning,
#                 "trust_score": trust_score,
#                 "committee_action": committee_action,
#                 "committee_confidence": committee_confidence,
#                 "expert_consensus": expert_consensus,
#                 "expert_confidence": expert_confidence,
#                 "regime": regime,
#                 "value_estimate": value_estimate,
#                 "gate_passed": bool(gate_passed),
#                 "position_size": position_size,
#             }

#             self.logger.info(
#                 f"[PPO ARBITER] {final_action.upper()} | gate={'PASS' if gate_passed else 'BLOCK'} | "
#                 f"conf={final_confidence:.2f} | size={position_size:.2%} | {reasoning}"
#             )

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO ARBITER] final_decision={final_decision}"
#                 )

#             return {
#                 "ppo_final_decision": final_decision,
#                 "ppo_gate_passed": bool(gate_passed),
#                 "ppo_position_size": position_size,
#             }

#         except Exception as e:
#             self.logger.error(f"[PPO ARBITER] Decision failed: {e}")
#             fallback = {
#                 "direction": "hold",
#                 "confidence": 0.0,
#                 "reasoning": f"Error in arbiter: {e}",
#                 "trust_score": 0.0,
#                 "committee_action": "hold",
#                 "committee_confidence": 0.0,
#                 "expert_consensus": "flat",
#                 "expert_confidence": 0.0,
#                 "regime": "unknown",
#                 "value_estimate": 0.0,
#                 "gate_passed": False,
#                 "position_size": 0.0,
#             }
#             return {
#                 "ppo_final_decision": fallback,
#                 "ppo_gate_passed": False,
#                 "ppo_position_size": 0.0,
#             }

#     def _build_arbiter_observation(
#         self,
#         committee: Dict[str, Any],
#         expert_signals: Dict[str, Any],
#         expert_consensus: str,
#         expert_confidence: float,
#         memory_gate: float,
#         fragility: float,
#     ) -> Dict[str, float]:
#         """
#         Build a small diagnostic observation for logging/interpretation.

#         NOTE: This is NOT used as NN input anymore; it is for introspection only.
#         The actual NN input is the unified 48-dim observation.
#         """
#         committee_action = committee.get("action", "hold")
#         committee_dir = (
#             1.0
#             if committee_action in ("long", "buy")
#             else (-1.0 if committee_action in ("short", "sell") else 0.0)
#         )
#         committee_conf = committee.get("confidence", 0.5)
#         consensus_score = committee.get("consensus_score", 0.5)

#         expert_dir = (
#             1.0
#             if expert_consensus in ("long", "buy")
#             else (-1.0 if expert_consensus in ("short", "sell") else 0.0)
#         )
#         agreement = 1.0 if (committee_dir * expert_dir > 0) else (
#             0.0 if expert_dir == 0 else -1.0
#         )

#         regime = expert_signals["market"].get("regime", "unknown")
#         regime_val = {
#             "trending": 0.8,
#             "mean_reverting": 0.5,
#             "volatile": -0.5,
#             "unknown": 0.0,
#         }.get(str(regime).lower(), 0.0)

#         return {
#             "committee_direction": committee_dir,
#             "committee_confidence": float(committee_conf),
#             "consensus_score": float(consensus_score),
#             "expert_direction": expert_dir,
#             "expert_confidence": float(expert_confidence),
#             "expert_committee_agreement": agreement,
#             "memory_gate": float(memory_gate),
#             "fragility": 1.0 - float(fragility),  # higher = better
#             "regime": regime_val,
#             "regime_strength": float(
#                 expert_signals["market"].get("regime_strength", 0.5)
#             ),
#         }

#     # ─────────────────────────────────────────────────────────────
#     # Voting interface (per-instrument proposals)
#     # ─────────────────────────────────────────────────────────────

#     async def vote(
#         self,
#         observation: Any = None,
#         action_vec: Optional[Union[List[float], np.ndarray]] = None,
#         thesis: Optional[str] = None,
#         **inputs,
#     ) -> Dict[str, Any]:
#         """
#         Produce per-instrument voting proposals.

#         PPOAgent is now an INFORMED decision-maker that:
#         1. Gathers expert signals (Trend, Momentum, Theme, Seasonality)
#         2. Considers risk signals and memory danger zones
#         3. Combines expert consensus with its own policy output

#         Returns dict with both per-instrument 'proposals' and legacy global fields.
#         """
#         try:
#             expert_signals = self._gather_expert_signals()
#             expert_consensus, expert_confidence = self._compute_expert_consensus(
#                 expert_signals
#             )

#             self.logger.debug(
#                 f"[PPO] Expert consensus: {expert_consensus} (conf={expert_confidence:.2f}) | "
#                 f"Regime: {expert_signals['market']['regime']} | "
#                 f"Memory gate: {expert_signals['memory'].get('memory_gate', 1.0)}"
#             )

#             # 1) Get an action vector from our policy if not provided
#             if action_vec is None:
#                 obs_vec = self._normalize_observation(observation)
#                 with torch.no_grad():
#                     action_mean, _, _ = self.network(
#                         torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)
#                     )
#                     action_vec = action_mean.squeeze(0).cpu().numpy()
#             else:
#                 action_vec = np.array(action_vec, dtype=np.float32)

#             # 2) Normalize to global direction/magnitude
#             action_arr = (
#                 action_vec
#                 if isinstance(action_vec, np.ndarray)
#                 else np.asarray(action_vec, dtype=np.float32)
#             )
#             policy_direction, policy_magnitude, raw_score = self._normalize_signal(
#                 action_arr
#             )

#             # 3) INFORMED DECISION: Blend policy with expert consensus
#             raw_memory_gate = expert_signals["memory"].get("memory_gate", 1.0)
#             if isinstance(raw_memory_gate, dict):
#                 raw_memory_gate = raw_memory_gate.get(
#                     "risk_multiplier", raw_memory_gate.get("value", 1.0)
#                 )
#             try:
#                 memory_gate = float(raw_memory_gate)
#             except (TypeError, ValueError):
#                 memory_gate = 1.0

#             if expert_confidence > 0.6 and policy_direction != expert_consensus:
#                 if expert_confidence > 0.75:
#                     global_direction = expert_consensus
#                     global_magnitude = policy_magnitude * 0.7
#                     self.logger.info(
#                         f"[PPO] Deferring to strong expert consensus: {expert_consensus}"
#                     )
#                 else:
#                     global_direction = "flat"
#                     global_magnitude = 0.3
#             else:
#                 global_direction = policy_direction
#                 global_magnitude = policy_magnitude
#                 if policy_direction == expert_consensus and expert_confidence > 0.5:
#                     global_magnitude = min(1.0, global_magnitude * 1.2)

#             # 4) Apply risk/memory gates
#             if memory_gate < 0.5:
#                 global_magnitude *= memory_gate
#                 self.logger.debug(
#                     f"[PPO] Memory gate reduced magnitude: {memory_gate:.2f}"
#                 )

#             danger_zones = expert_signals["memory"].get("danger_zones", {})
#             danger_zone_count = 0
#             if isinstance(danger_zones, dict):
#                 danger_zone_count = int(danger_zones.get("zone_count", 0))
#             elif isinstance(danger_zones, list):
#                 danger_zone_count = len(danger_zones)

#             if danger_zone_count > 0:
#                 global_magnitude *= 0.7
#                 self.logger.debug(
#                     f"[PPO] Danger zones detected ({danger_zone_count} zones), reducing magnitude"
#                 )

#             # 5) Base confidence informed by performance / exploration
#             conf_inputs = {"action": {"action": action_arr.tolist()}}
#             base_conf = await self.calculate_confidence(**conf_inputs)

#             if policy_direction == expert_consensus:
#                 base_conf = min(1.0, base_conf * (1.0 + expert_confidence * 0.3))
#             else:
#                 base_conf *= 0.7

#             if self.circuit_breaker["state"] == "OPEN":
#                 base_conf = float(max(0.05, base_conf * 0.5))
#             if self._health_status != "healthy":
#                 base_conf = float(max(0.1, base_conf * 0.7))
#             base_conf = float(np.clip(base_conf, 0.0, 1.0))

#             # 6) Get per-instrument market data
#             market_data = inputs.get("market_data") or self.smart_bus.get(
#                 "market_data", "PPOAgent"
#             ) or {}
#             price_data = self.smart_bus.get("price_data", "PPOAgent") or {}
#             indicators = self.smart_bus.get("technical_indicators", "PPOAgent") or {}

#             # 7) Generate per-instrument proposals
#             vote = PerInstrumentVote(member="PPOAgent")

#             for instrument in DEFAULT_INSTRUMENTS:
#                 inst_proposal = self._generate_instrument_proposal(
#                     instrument=instrument,
#                     global_direction=global_direction,
#                     global_magnitude=global_magnitude,
#                     base_conf=base_conf,
#                     market_data=market_data,
#                     price_data=price_data,
#                     indicators=indicators,
#                     thesis=thesis,
#                 )
#                 vote.set_proposal(inst_proposal)

#             payload = vote.to_dict()
#             payload["meta"] = {
#                 "avg_reward": self.training_stats.get("avg_episode_reward", 0.0),
#                 "explained_variance": self.training_stats.get(
#                     "explained_variance", 0.0
#                 ),
#                 "exploration_level": self.action_statistics.get(
#                     "exploration_level", 0.5
#                 ),
#                 "health": self._health_status,
#                 "circuit_breaker": self.circuit_breaker["state"],
#                 "raw_score": float(raw_score),
#                 "expert_consensus": expert_consensus,
#                 "expert_confidence": expert_confidence,
#                 "policy_direction": policy_direction,
#                 "aligned_with_experts": policy_direction == expert_consensus,
#             }
#             payload["rationale"] = thesis or (
#                 f"PPO informed decision (experts: {expert_consensus}, "
#                 f"policy: {policy_direction})"
#             )

#             return payload

#         except Exception as e:
#             self.logger.error(f"PPO vote() failed: {e}")
#             from modules.voting.core.per_instrument import create_flat_vote

#             error_vote = create_flat_vote("PPOAgent")
#             payload = error_vote.to_dict()
#             payload["rationale"] = f"Vote fallback due to error: {e}"
#             return payload

#     def _generate_instrument_proposal(
#         self,
#         instrument: str,
#         global_direction: str,
#         global_magnitude: float,
#         base_conf: float,
#         market_data: Dict[str, Any],
#         price_data: Dict[str, Any],
#         indicators: Dict[str, Any],
#         thesis: Optional[str] = None,
#     ) -> InstrumentProposal:
#         """
#         Generate a voting proposal for a specific instrument.

#         Combines global policy direction with instrument-specific market analysis.
#         """
#         try:
#             inst_market = extract_instrument_data(market_data, instrument)
#             inst_price = extract_instrument_data(price_data, instrument)
#             inst_indicators = extract_instrument_data(indicators, instrument)

#             inst_trend, inst_strength = analyze_instrument_trend(
#                 inst_price, inst_indicators
#             )

#             if global_direction == "flat":
#                 if inst_strength > 0.3:
#                     final_direction = inst_trend
#                     final_confidence = base_conf * inst_strength
#                     final_magnitude = inst_strength
#                 else:
#                     final_direction = "flat"
#                     final_confidence = base_conf * 0.3
#                     final_magnitude = 0.0
#             elif inst_trend == global_direction:
#                 final_direction = global_direction
#                 final_confidence = min(
#                     1.0, base_conf * (1.0 + inst_strength * 0.3)
#                 )
#                 final_magnitude = min(
#                     1.0, global_magnitude * (1.0 + inst_strength * 0.2)
#                 )
#             elif inst_trend == "flat":
#                 final_direction = global_direction
#                 final_confidence = base_conf * 0.7
#                 final_magnitude = global_magnitude * 0.8
#             else:
#                 final_direction = "flat"
#                 final_confidence = base_conf * 0.3
#                 final_magnitude = 0.0

#             return InstrumentProposal(
#                 instrument=instrument,
#                 action=final_direction,
#                 confidence=round(final_confidence, 4),
#                 magnitude=round(final_magnitude, 4),
#                 horizon="intraday",
#                 rationale=(
#                     thesis
#                     or f"PPO: global={global_direction}, {instrument}_trend={inst_trend}, "
#                     f"strength={inst_strength:.2f}"
#                 ),
#                 meta={
#                     "global_direction": global_direction,
#                     "instrument_trend": inst_trend,
#                     "instrument_strength": inst_strength,
#                 },
#             )

#         except Exception as e:
#             self.logger.warning(f"Failed to generate proposal for {instrument}: {e}")
#             return InstrumentProposal(
#                 instrument=instrument,
#                 action="flat",
#                 confidence=0.2,
#                 magnitude=0.0,
#                 rationale=f"Fallback: {e}",
#             )

#     def _normalize_signal(
#         self, action_vec: Optional[Union[List[float], np.ndarray]]
#     ) -> Tuple[str, float, float]:
#         """
#         Map a continuous action vector to (direction, magnitude, raw_score).
#         Uses hysteresis to prevent flip-flopping between long/short.

#         Heuristic:
#           - raw_score = mean(action_vec)
#           - direction = sign(raw_score) with deadband + hysteresis
#           - magnitude = clipped L2 norm scaled by vector length
#         """
#         if action_vec is None or len(action_vec) == 0:
#             return "flat", 0.0, 0.0

#         action_vec = np.array(action_vec, dtype=np.float32).reshape(-1)
#         raw_score = float(np.mean(action_vec))

#         entry_threshold = 0.10
#         reversal_threshold = 0.15
#         exit_threshold = 0.03

#         last_dir = getattr(self, "_last_direction", "flat")

#         if last_dir == "flat":
#             if raw_score > entry_threshold:
#                 direction = "long"
#             elif raw_score < -entry_threshold:
#                 direction = "short"
#             else:
#                 direction = "flat"
#         elif last_dir == "long":
#             if raw_score < -reversal_threshold:
#                 direction = "short"
#             elif raw_score < -exit_threshold:
#                 direction = "flat"
#             else:
#                 direction = "long"
#         else:  # last_dir == 'short'
#             if raw_score > reversal_threshold:
#                 direction = "long"
#             elif raw_score > exit_threshold:
#                 direction = "flat"
#             else:
#                 direction = "short"

#         if direction != last_dir:
#             self._direction_hold_count = 0
#         else:
#             self._direction_hold_count = getattr(
#                 self, "_direction_hold_count", 0
#             ) + 1
#         self._last_direction = direction

#         norm = float(np.linalg.norm(action_vec))
#         scale = max(1.0, np.sqrt(len(action_vec))) * 2.0
#         magnitude = float(np.clip(norm / scale, 0.0, 1.0))
#         if direction == "flat":
#             magnitude = 0.0

#         if self.debug:
#             self.logger.debug(
#                 f"[DEBUG][PPO] _normalize_signal: raw_score={raw_score:.3f}, "
#                 f"direction={direction}, magnitude={magnitude:.3f}, "
#                 f"last_direction={last_dir}, hold_count={self._direction_hold_count}"
#             )

#         return direction, magnitude, raw_score

#     # ─────────────────────────────────────────────────────────────
#     # Legacy / utility methods
#     # ─────────────────────────────────────────────────────────────

#     def record_step(
#         self, obs_vec: np.ndarray, reward: float, done: bool = False, **kwargs
#     ):
#         """Legacy compatibility for step recording (training env)."""
#         experience = {"observation": obs_vec, "reward": reward, "done": done}

#         if hasattr(self, "last_action"):
#             experience["action"] = self.last_action
#         if hasattr(self, "_last_log_prob"):
#             experience["log_prob"] = self._last_log_prob
#         if hasattr(self, "_last_value"):
#             experience["value"] = self._last_value

#         for key, value in experience.items():
#             if key == "observation":
#                 self.buffer["observations"].append(value)
#             elif key == "action":
#                 self.buffer["actions"].append(value)
#             elif key == "reward":
#                 self.buffer["rewards"].append(value)
#             elif key == "log_prob":
#                 self.buffer["log_probs"].append(value)
#             elif key == "value":
#                 self.buffer["values"].append(value)
#             elif key == "done":
#                 self.buffer["dones"].append(value)

#     def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         Legacy compatibility for action selection (synchronous API).

#         This uses the same forward-pass logic as _process_action_selection,
#         but without the async plumbing.
#         """
#         try:
#             obs_np = obs_tensor.cpu().numpy()
#             obs_vec = self._normalize_observation(obs_np)
#             self._last_obs_vec = obs_vec.copy()
#             obs = torch.from_numpy(obs_vec).to(self.device).unsqueeze(0)

#             with torch.no_grad():
#                 action_mean, action_log_std, value = self.network(obs)
#                 action_std = torch.exp(action_log_std)
#                 dist = torch.distributions.Normal(action_mean, action_std)
#                 action = dist.sample()
#                 log_prob = dist.log_prob(action).sum(dim=-1)

#             action_np = action.squeeze().cpu().numpy()
#             self.last_action = action_np
#             self._last_log_prob = float(log_prob.item())
#             self._last_value = float(value.squeeze().cpu().item())

#             if self.debug:
#                 self.logger.debug(
#                     f"[DEBUG][PPO] select_action: action={action_np.tolist()}, "
#                     f"value_est={self._last_value:.4f}"
#                 )

#             return torch.tensor(action_np, dtype=torch.float32)

#         except Exception as e:
#             self.logger.error(f"PPO select_action failed: {e}")
#             return torch.zeros(self._cfg.act_size, dtype=torch.float32)

#     def end_episode(self, **kwargs):
#         """Legacy compatibility for episode end (training env)."""
#         if self.buffer["rewards"]:
#             episode_reward = float(sum(self.buffer["rewards"]))
#             self.episode_rewards.append(episode_reward)
#             self.training_stats["episodes_completed"] += 1

#             if episode_reward > self.training_stats["best_episode_reward"]:
#                 self.training_stats["best_episode_reward"] = episode_reward

#     def get_state(self) -> Dict[str, Any]:
#         """Get module state for persistence."""
#         return {
#             "training_stats": self.training_stats.copy(),
#             "action_statistics": self.action_statistics.copy(),
#             "genome": self.genome.copy(),
#             "episodes_completed": self.training_stats["episodes_completed"],
#             "total_updates": self.training_stats["total_updates"],
#             "best_performance": self.best_performance,
#             "circuit_breaker": self.circuit_breaker.copy(),
#             "health_status": self._health_status,
#             "network_state": self.network.state_dict()
#             if hasattr(self, "network")
#             else {},
#             "optimizer_state": self.optimizer.state_dict()
#             if hasattr(self, "optimizer")
#             else {},
#         }

#     def set_state(self, state: Dict[str, Any]):
#         """Set module state from persistence."""
#         if "training_stats" in state:
#             self.training_stats.update(state["training_stats"])

#         if "action_statistics" in state:
#             self.action_statistics.update(state["action_statistics"])

#         if "genome" in state:
#             self.genome.update(state["genome"])

#         if "best_performance" in state:
#             self.best_performance = state["best_performance"]

#         if "circuit_breaker" in state:
#             self.circuit_breaker.update(state["circuit_breaker"])

#         if "health_status" in state:
#             self._health_status = state["health_status"]

#         if "network_state" in state and hasattr(self, "network"):
#             try:
#                 self.network.load_state_dict(state["network_state"])
#             except Exception as e:
#                 self.logger.warning(f"Failed to restore network state: {e}")

#         if "optimizer_state" in state and hasattr(self, "optimizer"):
#             try:
#                 self.optimizer.load_state_dict(state["optimizer_state"])
#             except Exception as e:
#                 self.logger.warning(f"Failed to restore optimizer state: {e}")

#     def get_health_status(self) -> Dict[str, Any]:
#         """Get health status summary."""
#         return {
#             "status": self._health_status,
#             "last_check": self._last_health_check,
#             "circuit_breaker": self.circuit_breaker["state"],
#             "episodes_completed": self.training_stats["episodes_completed"],
#             "average_reward": self.training_stats["avg_episode_reward"],
#             "learning_rate": self.training_stats["learning_rate"],
#         }

#     def stop_monitoring(self):
#         """Stop background monitoring thread."""
#         self._monitoring_active = False

#     def confidence(self, obs: Any = None, **kwargs) -> float:
#         """Legacy compatibility for confidence scoring."""
#         performance_confidence = max(
#             0.0, min(1.0, (self.training_stats["avg_episode_reward"] + 50.0) / 100.0)
#         )
#         exploration_confidence = 1.0 - self.action_statistics["exploration_level"]

#         return float((performance_confidence + exploration_confidence) / 2.0)

#     # ─────────────────────────────────────────────────────────────
#     # END PPOAgent
#     # ─────────────────────────────────────────────────────────────
