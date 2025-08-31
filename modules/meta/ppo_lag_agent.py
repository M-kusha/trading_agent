# ─────────────────────────────────────────────────────────────
# File: modules/meta/ppo_lag_agent.py
# [ROCKET] PRODUCTION-READY PPO-Lag Agent System
# Enhanced with SmartInfoBus integration & advanced market adaptation
# v2.7 — Pylance-clean (config typing fixed), safer math, contract-stable payloads
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import time
import threading
from modules.contracts import module_args
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────

class MarketAwarePPONetwork(nn.Module):
    """Advanced PPO network with market-aware architecture"""

    def __init__(self, obs_size: int, act_size: int, lag_window: int, hidden_size: int = 128):
        super().__init__()

        self.obs_size = int(obs_size)
        self.act_size = int(act_size)
        self.lag_window = int(lag_window)
        self.lag_features = 4  # returns, volatility, volume, spread

        # Calculate extended observation size
        self.extended_obs_size = self.obs_size + (self.lag_window * self.lag_features) + 6  # +6 for position features

        # Observation preprocessing
        self.obs_normalizer = nn.BatchNorm1d(self.extended_obs_size, momentum=0.1)

        # Market context encoder
        self.market_encoder = nn.Sequential(
            nn.Linear(self.lag_window * self.lag_features, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # Main feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.extended_obs_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Market-aware actor with attention
        self.actor_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )

        self.actor_head = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, self.act_size),
            nn.Tanh()
        )

        # Dual critic heads for improved value estimation
        self.value_head_1 = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        self.value_head_2 = nn.Sequential(
            nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )

        # Adaptive action std with market conditioning
        self.log_std_base = nn.Parameter(torch.log(torch.ones(self.act_size) * 0.1))
        self.std_conditioner = nn.Linear(hidden_size // 2, self.act_size)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Advanced weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor, market_lags: torch.Tensor):
        """Enhanced forward pass with market adaptation"""
        try:
            # Validate inputs
            if torch.any(torch.isnan(obs)):
                obs = torch.nan_to_num(obs)
            if torch.any(torch.isnan(market_lags)):
                market_lags = torch.nan_to_num(market_lags)

            batch_size = obs.shape[0]

            # Normalize observations
            if batch_size > 1:
                obs_norm = self.obs_normalizer(obs)
            else:
                obs_norm = obs  # Skip batch norm for single samples

            # Extract market context
            market_features = self.market_encoder(market_lags)

            # Main feature extraction
            main_features = self.feature_extractor(obs_norm)

            # Apply attention for market-aware processing
            main_features_expanded = main_features.unsqueeze(1)  # Add sequence dimension
            attended_features, _ = self.actor_attention(
                main_features_expanded, main_features_expanded, main_features_expanded
            )
            attended_features = attended_features.squeeze(1)  # Remove sequence dimension

            # Combine features
            combined_features = torch.cat([attended_features, market_features], dim=-1)

            # Actor output with market-conditioned std
            action_mean = self.actor_head(combined_features)

            # Adaptive standard deviation based on market conditions
            std_adjustment = torch.sigmoid(self.std_conditioner(market_features))
            action_std = torch.exp(self.log_std_base) * (0.5 + std_adjustment)
            action_std = torch.clamp(action_std, 0.01, 1.0)

            # Dual critic values
            value_1 = self.value_head_1(combined_features)
            value_2 = self.value_head_2(combined_features)
            value = torch.min(value_1, value_2)  # Conservative value estimate

            # Validate outputs
            if torch.any(torch.isnan(action_mean)):
                action_mean = torch.zeros_like(action_mean)
            if torch.any(torch.isnan(action_std)):
                action_std = torch.ones_like(action_std) * 0.1
            if torch.any(torch.isnan(value)):
                value = torch.zeros_like(value)

            return action_mean, action_std, value

        except Exception:
            # Safe fallback
            batch_size = obs.shape[0] if obs.ndim > 1 else 1
            device = obs.device
            return (
                torch.zeros(batch_size, self.act_size, device=device),
                torch.ones(batch_size, self.act_size, device=device) * 0.1,
                torch.zeros(batch_size, 1, device=device)
            )


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────

@dataclass
class PPOLagConfig:
    """Configuration for PPO-Lag Agent"""
    obs_size: int = 64
    act_size: int = 2
    hidden_size: int = 128
    lr: float = 1e-4
    lag_window: int = 20
    adv_decay: float = 0.95
    vol_scaling: bool = True
    position_aware: bool = True
    device: str = "cpu"

    # Performance thresholds
    max_processing_time_ms: float = 300
    circuit_breaker_threshold: int = 3
    min_episode_length: int = 10

    # PPO parameters
    clip_eps: float = 0.1
    value_coeff: float = 0.5
    entropy_coeff: float = 0.001
    gae_lambda: float = 0.95
    gamma: float = 0.99
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    batch_size: int = 64
    target_kl: float = 0.01


# ─────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────

@module(**module_args(
    "PPOLagAgent",
    description="Advanced PPO-Lag agent with market adaptation and SmartInfoBus integration",
    error_handling=True,
    hot_reload=True,
    timeout_ms=120,
))
class PPOLagAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced PPO-Lag agent with SmartInfoBus integration.
    Incorporates lagged market features and intelligent position awareness for trading.
    """

    # Keep BaseModule.config as Dict[str, Any] for framework compatibility.
    # Use a typed twin self.cfg: PPOLagConfig for attribute-safe access.
    cfg: PPOLagConfig
    config: Dict[str, Any]

    def __init__(
        self,
        config: Optional[PPOLagConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Minimal pre-init so BaseModule (which may call _initialize early) doesn't crash
        self._preinitialize_minimum()
        super().__init__()

        # 1) Typed config holder + dict mirror for BaseModule
        self.cfg = config if config is not None else PPOLagConfig()
        self.config = asdict(self.cfg)

        # 2) Initialize advanced systems
        self._initialize_advanced_systems()

        # 3) Initialize genome parameters
        self._initialize_genome_parameters(genome)

        # 4) Initialize agent state
        self._initialize_agent_state()

        # 5) Start background monitoring after init is complete
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                "[TARGET]", "PPO_LAG_AGENT_INITIALIZED",
                details=f"Obs size: {self.cfg.obs_size}, Lag window: {self.cfg.lag_window}",
                result="Market-aware PPO agent ready",
                context="ppo_lag_training"
            )
        )

    # ─────────────────────────────────────────────────────────
    # Helpers: config sync
    # ─────────────────────────────────────────────────────────
    def _preinitialize_minimum(self) -> None:
        """Ensure core attributes exist before BaseModule may trigger _initialize early."""
        # Smart bus
        if not hasattr(self, 'smart_bus') or getattr(self, 'smart_bus', None) is None:
            try:
                self.smart_bus = InfoBusManager.get_instance()
            except Exception:
                # Leave unset on failure; guards will skip bus writes until available
                pass

        # Logger (fallback stub if RotatingLogger cannot be created yet)
        if not hasattr(self, 'logger') or getattr(self, 'logger', None) is None:
            try:
                self.logger = RotatingLogger(
                    name="PPOLagAgent",
                    log_path="logs/meta/ppo_lag_agent.log",
                    max_lines=3000,
                    operator_mode=True,
                    plain_english=True
                )
            except Exception:
                class _StubLogger:
                    def info(self, *args, **kwargs):
                        pass
                    def warning(self, *args, **kwargs):
                        pass
                    def error(self, *args, **kwargs):
                        pass
                    def debug(self, *args, **kwargs):
                        pass
                self.logger = _StubLogger()

    def _sync_config_dict(self) -> None:
        """Keep BaseModule.config mirror in sync with self.cfg."""
        self.config.update(asdict(self.cfg))

    def _update_cfg_from_dict(self, d: Dict[str, Any]) -> None:
        """Update self.cfg fields from a dict (safe & partial)."""
        for k, v in d.items():
            if hasattr(self.cfg, k):
                setattr(self.cfg, k, v)
        self._sync_config_dict()

    # ─────────────────────────────────────────────────────────
    # Systems, genome, state
    # ─────────────────────────────────────────────────────────
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for PPO-Lag agent"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PPOLagAgent",
            log_path="logs/meta/ppo_lag_agent.log",
            max_lines=3000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PPOLagAgent", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for training operations
        self.circuit_breaker: Dict[str, Any] = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': int(self.cfg.circuit_breaker_threshold)
        }

        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome: Dict[str, Any] = {
                "obs_size": int(genome.get("obs_size", self.cfg.obs_size)),
                "act_size": int(genome.get("act_size", self.cfg.act_size)),
                "hidden_size": int(genome.get("hidden_size", self.cfg.hidden_size)),
                "lr": float(genome.get("lr", self.cfg.lr)),
                "lag_window": int(genome.get("lag_window", self.cfg.lag_window)),
                "adv_decay": float(genome.get("adv_decay", self.cfg.adv_decay)),
                "vol_scaling": bool(genome.get("vol_scaling", self.cfg.vol_scaling)),
                "position_aware": bool(genome.get("position_aware", self.cfg.position_aware))
            }
        else:
            self.genome = {
                "obs_size": self.cfg.obs_size,
                "act_size": self.cfg.act_size,
                "hidden_size": self.cfg.hidden_size,
                "lr": self.cfg.lr,
                "lag_window": self.cfg.lag_window,
                "adv_decay": self.cfg.adv_decay,
                "vol_scaling": self.cfg.vol_scaling,
                "position_aware": self.cfg.position_aware
            }

    def _initialize_agent_state(self):
        """Initialize PPO-Lag agent state"""
        # Core parameters
        self.device = torch.device(self.cfg.device)

        # Enhanced network architecture
        self.network = MarketAwarePPONetwork(
            int(self.genome["obs_size"]),
            int(self.genome["act_size"]),
            int(self.genome["lag_window"]),
            int(self.genome["hidden_size"])
        ).to(self.device)

        # Separate optimizers for different components
        self.actor_optimizer = optim.Adam(
            list(self.network.feature_extractor.parameters()) +
            list(self.network.actor_attention.parameters()) +
            list(self.network.actor_head.parameters()) +
            [self.network.log_std_base] +
            list(self.network.std_conditioner.parameters()),
            lr=float(self.genome["lr"]), eps=1e-5
        )

        self.critic_optimizer = optim.Adam(
            list(self.network.market_encoder.parameters()) +
            list(self.network.value_head_1.parameters()) +
            list(self.network.value_head_2.parameters()),
            lr=float(self.genome["lr"]) * 2.0, eps=1e-5  # Higher LR for critics
        )

        # Market adaptation state
        self.market_regime_adaptation: Dict[str, Dict[str, float]] = {
            'trending': {'std_multiplier': 1.2, 'clip_adjustment': 0.0},
            'volatile': {'std_multiplier': 0.8, 'clip_adjustment': 0.05},
            'ranging': {'std_multiplier': 1.0, 'clip_adjustment': -0.02},
            'unknown': {'std_multiplier': 1.0, 'clip_adjustment': 0.0}
        }

        # Lag buffers for market features
        self.price_buffer: Deque[float] = deque(maxlen=int(self.genome["lag_window"]))
        self.volume_buffer: Deque[float] = deque(maxlen=int(self.genome["lag_window"]))
        self.spread_buffer: Deque[float] = deque(maxlen=int(self.genome["lag_window"]))
        self.volatility_buffer: Deque[float] = deque(maxlen=int(self.genome["lag_window"]))

        # Experience buffer
        self.buffer: Dict[str, List[Any]] = {
            'observations': [],
            'market_features': [],
            'actions': [],
            'log_probs': [],
            'values': [],
            'rewards': [],
            'advantages': [],
            'returns': [],
            'dones': []
        }

        # Advanced tracking
        self.episode_rewards: Deque[float] = deque(maxlen=200)
        self.episode_lengths: Deque[int] = deque(maxlen=200)
        self.market_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})
        self.volatility_performance: Dict[str, Dict[str, Any]] = defaultdict(lambda: {'rewards': deque(maxlen=100), 'count': 0})

        # Position and risk tracking
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.position_history: Deque[Dict[str, Any]] = deque(maxlen=1000)
        self.risk_metrics: Dict[str, float] = {
            'max_position': 0.0,
            'avg_position': 0.0,
            'position_volatility': 0.0,
            'risk_adjusted_return': 0.0
        }

        # Training statistics
        self.training_stats: Dict[str, Any] = {
            'total_updates': 0,
            'episodes_completed': 0,
            'actor_loss_trend': 0.0,
            'critic_loss_trend': 0.0,
            'kl_divergence': 0.0,
            'explained_variance': 0.0,
            'policy_entropy': 0.0,
            'advantage_mean': 0.0,
            'advantage_std': 0.0
        }

        # Adaptive parameters
        self.running_adv_std = 1.0
        self.adaptive_clip_eps = float(self.cfg.clip_eps)
        self.adaptive_lr_factor = 1.0

    # ─────────────────────────────────────────────────────────
    # Monitoring
    # ─────────────────────────────────────────────────────────
    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_agent_health()
                    self._analyze_agent_performance()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")

        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self):
        """Initialize module (framework calls)"""
        try:
            # Guard: ensure smart_bus/logger exist even on early calls
            if not hasattr(self, 'smart_bus') or getattr(self, 'smart_bus', None) is None:
                try:
                    self.smart_bus = InfoBusManager.get_instance()
                except Exception:
                    pass
            if not hasattr(self, 'logger') or getattr(self, 'logger', None) is None:
                try:
                    self.logger = RotatingLogger(
                        name="PPOLagAgent",
                        log_path="logs/meta/ppo_lag_agent.log",
                        max_lines=3000,
                        operator_mode=True,
                        plain_english=True
                    )
                except Exception:
                    class _StubLogger:
                        def info(self, *args, **kwargs):
                            pass
                        def warning(self, *args, **kwargs):
                            pass
                        def error(self, *args, **kwargs):
                            pass
                        def debug(self, *args, **kwargs):
                            pass
                    self.logger = _StubLogger()

            # Set initial agent status in SmartInfoBus
            net_params = 0
            try:
                if hasattr(self, 'network') and self.network is not None:
                    net_params = int(sum(p.numel() for p in self.network.parameters()))
            except Exception:
                net_params = 0
            initial_status = {
                "agent_type": "ppo-lag",
                "episodes_completed": 0,
                "training_active": False,
                "network_parameters": net_params
            }

            sb = getattr(self, 'smart_bus', None)
            if sb is not None:
                sb.set(
                    'agent_status',
                    initial_status,
                    module='PPOLagAgent',
                    thesis="Initial PPO-Lag agent status"
                )
            else:
                # Log only if logger exists; skip bus write until later
                self.logger.warning("SmartInfoBus not available during early _initialize; will set status later")

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")

    # ─────────────────────────────────────────────────────────
    # Main process loop
    # ─────────────────────────────────────────────────────────
    async def process(self, **inputs) -> Dict[str, Any]:
        """Process PPO-Lag agent operations"""
        start_time = time.time()

        try:
            # Extract agent data
            agent_data = await self._extract_agent_data(**inputs)

            if not agent_data:
                return await self._handle_no_data_fallback()

            # Update market buffers
            market_result = await self._update_market_buffers(agent_data)

            # Process training step if data available
            training_result = await self._process_training_step(agent_data)
            market_result.update(training_result)

            # Adapt to market conditions
            adaptation_result = await self._adapt_to_market_conditions(agent_data)
            market_result.update(adaptation_result)

            # Update performance tracking
            performance_result = await self._update_performance_tracking(agent_data)
            market_result.update(performance_result)

            # Generate thesis
            thesis = await self._generate_agent_thesis(agent_data, market_result)

            # Update SmartInfoBus
            await self._update_agent_smart_bus(market_result, thesis)

            # Record success
            processing_time = (time.time() - start_time) * 1000.0
            self._record_success(processing_time)

            # Contract-stable payload
            agent_status = {
                'agent_type': 'ppo-lag',
                'episodes_completed': self.training_stats['episodes_completed'],
                'total_updates': self.training_stats['total_updates'],
                'network_parameters': int(sum(p.numel() for p in self.network.parameters())),
                'training_active': len(self.buffer['rewards']) > 0,
                'buffer_size': len(self.buffer['rewards'])
            }
            training_metrics = {
                'training_stats': dict(self.training_stats),
                'episode_rewards': list(self.episode_rewards)[-10:],
                'episode_lengths': list(self.episode_lengths)[-10:],
                'adaptive_parameters': {
                    'clip_eps': self.adaptive_clip_eps,
                    'lr_factor': self.adaptive_lr_factor,
                    'entropy_coeff': self.cfg.entropy_coeff
                }
            }
            position_metrics = {
                'current_position': float(self.position),
                'unrealized_pnl': float(self.unrealized_pnl),
                'risk_metrics': dict(self.risk_metrics),
                'position_history_size': len(self.position_history)
            }
            market_adaptation = {
                'lag_window': int(self.genome["lag_window"]),
                'buffer_sizes': {
                    'price': len(self.price_buffer),
                    'volume': len(self.volume_buffer),
                    'spread': len(self.spread_buffer),
                    'volatility': len(self.volatility_buffer)
                },
                'market_performance': {
                    regime: {
                        'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                        'count': int(data['count'])
                    } for regime, data in self.market_performance.items()
                },
                'volatility_performance': {
                    vol_level: {
                        'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                        'count': int(data['count'])
                    } for vol_level, data in self.volatility_performance.items()
                },
                'adaptation_settings': {
                    'vol_scaling': bool(self.genome["vol_scaling"]),
                    'position_aware': bool(self.genome["position_aware"])
                }
            }
            market_result.update({
                'agent_status': agent_status,
                'ppo_lag_training_metrics': training_metrics,
                'position_metrics': position_metrics,
                'market_adaptation': market_adaptation,
                '_thesis': thesis
            })

            return market_result

        except Exception as e:
            return await self._handle_agent_error(e, start_time)

    # ─────────────────────────────────────────────────────────
    # Data extraction & buffers
    # ─────────────────────────────────────────────────────────
    async def _extract_agent_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract agent data from SmartInfoBus"""
        try:
            trades = self.smart_bus.get('trades', 'PPOLagAgent') or []
            actions = self.smart_bus.get('actions', 'PPOLagAgent') or []
            market_data = self.smart_bus.get('market_data', 'PPOLagAgent') or {}
            # Backfill missing fields from canonical bus keys
            if isinstance(market_data, dict):
                if not market_data.get('regime'):
                    regime_fallback = (
                        self.smart_bus.get('market_regime', 'PPOLagAgent')
                        or (self.smart_bus.get('market_state', 'PPOLagAgent') or {}).get('regime')
                        or (self.smart_bus.get('regime_prediction', 'PPOLagAgent') or {}).get('predicted')
                    )
                    if regime_fallback:
                        market_data['regime'] = regime_fallback
                if not market_data.get('session'):
                    session_fallback = (
                        self.smart_bus.get('trading_session', 'PPOLagAgent')
                        or (self.smart_bus.get('market_context', 'PPOLagAgent') or {}).get('session')
                        or (self.smart_bus.get('market_context', 'PPOLagAgent') or {}).get('session_canonical')
                    )
                    if session_fallback:
                        market_data['session'] = session_fallback
                if not market_data.get('volatility_level'):
                    ms = self.smart_bus.get('market_state', 'PPOLagAgent') or {}
                    vol_val = ms.get('volatility')
                    if isinstance(vol_val, (int, float)):
                        market_data['volatility_level'] = 'high' if vol_val >= 0.7 else ('medium' if vol_val >= 0.35 else 'low')
            training_signals = self.smart_bus.get('training_signals', 'PPOLagAgent') or {}

            context = self._extract_standard_context(market_data)

            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'training_signals': training_signals,
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'obs_vec': inputs.get('obs_vec', None),
                'reward': inputs.get('reward', None),
                'done': bool(inputs.get('done', False))
            }

        except Exception as e:
            self.logger.error(f"Failed to extract agent data: {e}")
            return None

    def _extract_standard_context(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract standard market context with robust fallbacks from InfoBus."""
        md = market_data or {}
        regime = md.get('regime') or md.get('market_regime')
        if not regime:
            regime = (
                self.smart_bus.get('market_regime', 'PPOLagAgent')
                or (self.smart_bus.get('market_state', 'PPOLagAgent') or {}).get('regime')
            ) or 'unknown'
        session = md.get('session') or (
            self.smart_bus.get('trading_session', 'PPOLagAgent')
            or (self.smart_bus.get('market_context', 'PPOLagAgent') or {}).get('session')
            or (self.smart_bus.get('market_context', 'PPOLagAgent') or {}).get('session_canonical')
        ) or 'unknown'
        vol_level = md.get('volatility_level') or 'medium'
        volatility = md.get('volatility')
        if volatility is None:
            ms = self.smart_bus.get('market_state', 'PPOLagAgent') or {}
            volatility = ms.get('volatility', 1.0)
            if 'volatility_level' not in md and isinstance(volatility, (int, float)):
                vol_level = 'high' if volatility >= 0.7 else ('medium' if volatility >= 0.35 else 'low')
        return {
            'regime': regime,
            'volatility_level': vol_level,
            'session': session,
            'price': float(md.get('price', 0.0) or 0.0),
            'volume': float(md.get('volume', 0.0) or 0.0),
            'spread': float(md.get('spread', 0.0) or 0.0),
            'volatility': float(volatility if isinstance(volatility, (int, float)) else (md.get('volatility', 1.0) or 1.0)),
            'timestamp': datetime.now().isoformat()
        }

    async def _update_market_buffers(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market buffers from agent data"""
        try:
            context = agent_data['context']

            current_price = float(context.get('price', 0.0) or 0.0)
            current_volume = float(context.get('volume', 0.0) or 0.0)
            current_spread = float(context.get('spread', 0.0) or 0.0)
            current_volatility = float(context.get('volatility', 1.0) or 1.0)

            self.update_market_buffers(current_price, current_volume, current_spread, current_volatility)

            return {
                'market_buffers_updated': True,
                'buffer_sizes': {
                    'price': len(self.price_buffer),
                    'volume': len(self.volume_buffer),
                    'spread': len(self.spread_buffer),
                    'volatility': len(self.volatility_buffer)
                }
            }

        except Exception as e:
            self.logger.error(f"Market buffer update failed: {e}")
            return {'market_buffers_updated': False, 'error': str(e)}

    def update_market_buffers(self, price: float, volume: float, spread: float, volatility: float):
        """Enhanced market buffer update with validation"""
        try:
            # Validate inputs
            if np.isnan(price) or price <= 0:
                price = self.price_buffer[-1] if self.price_buffer else 1.0
            if np.isnan(volume) or volume < 0:
                volume = 0.0
            if np.isnan(spread) or spread < 0:
                spread = 0.0
            if np.isnan(volatility) or volatility <= 0:
                volatility = 1.0

            # Calculate returns
            if len(self.price_buffer) > 0:
                last_price = self.price_buffer[-1]
                if last_price != 0:
                    price_return = (price - last_price) / max(abs(last_price), 1e-9)
                else:
                    price_return = 0.0
            else:
                price_return = 0.0

            # Validate and clamp return
            if np.isnan(price_return) or abs(price_return) > 0.1:  # Cap at 10%
                price_return = 0.0

            # Update buffers
            self.price_buffer.append(float(price_return))
            self.volume_buffer.append(float(volume))
            self.spread_buffer.append(float(spread))
            self.volatility_buffer.append(float(volatility))

        except Exception as e:
            self.logger.error(f"Buffer update failed: {e}")

    def get_lag_features(self) -> np.ndarray:
        """Extract enhanced lagged features"""
        try:
            L = int(self.genome["lag_window"])
            price_lags = list(self.price_buffer) + [0.0] * (L - len(self.price_buffer))
            volume_lags = list(self.volume_buffer) + [0.0] * (L - len(self.volume_buffer))
            spread_lags = list(self.spread_buffer) + [0.0] * (L - len(self.spread_buffer))
            vol_lags = list(self.volatility_buffer) + [1.0] * (L - len(self.volatility_buffer))

            # Interleave features for better temporal representation
            features: List[float] = []
            for i in range(L):
                features.extend([
                    float(price_lags[i]),
                    float(vol_lags[i]),
                    float(volume_lags[i]),
                    float(spread_lags[i])
                ])

            result = np.array(features, dtype=np.float32)

            # Validate and normalize features
            if np.any(np.isnan(result)):
                result = np.nan_to_num(result)

            # Apply feature scaling
            result = np.clip(result, -5.0, 5.0)

            return result

        except Exception as e:
            self.logger.error(f"Lag feature extraction failed: {e}")
            return np.zeros(int(self.genome["lag_window"]) * 4, dtype=np.float32)

    # ─────────────────────────────────────────────────────────
    # Adaptation & training
    # ─────────────────────────────────────────────────────────
    async def _adapt_to_market_conditions(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced market condition adaptation"""
        try:
            context = agent_data['context']
            regime = context.get('regime', 'unknown')
            vol_level = context.get('volatility_level', 'medium')

            # Get adaptation parameters
            adaptation = self.market_regime_adaptation.get(regime, self.market_regime_adaptation['unknown'])

            # Adapt clipping epsilon
            base_clip = float(self.cfg.clip_eps)
            self.adaptive_clip_eps = float(np.clip(base_clip + adaptation['clip_adjustment'], 0.05, 0.3))

            # Adapt learning rate
            if regime == 'volatile':
                self.adaptive_lr_factor = 0.8  # Slower learning in volatile markets
            elif regime == 'trending':
                self.adaptive_lr_factor = 1.2  # Faster learning in trending markets
            else:
                self.adaptive_lr_factor = 1.0

            # Update optimizer learning rates
            for optimizer in [self.actor_optimizer, self.critic_optimizer]:
                for param_group in optimizer.param_groups:
                    base_lr = float(self.genome["lr"]) if optimizer == self.actor_optimizer else float(self.genome["lr"]) * 2.0
                    param_group['lr'] = base_lr * self.adaptive_lr_factor

            # Adapt entropy coefficient based on volatility
            if vol_level == 'extreme':
                self.cfg.entropy_coeff = min(0.005, float(self.cfg.entropy_coeff) * 0.9)  # Reduce exploration
            elif vol_level == 'low':
                self.cfg.entropy_coeff = min(0.02, float(self.cfg.entropy_coeff) * 1.05)  # Increase exploration
            self._sync_config_dict()

            return {
                'market_adapted': True,
                'regime': regime,
                'vol_level': vol_level,
                'adaptive_clip': self.adaptive_clip_eps,
                'lr_factor': self.adaptive_lr_factor,
                'entropy_coeff': self.cfg.entropy_coeff
            }

        except Exception as e:
            self.logger.error(f"Market adaptation failed: {e}")
            return {'market_adapted': False, 'error': str(e)}

    async def _process_training_step(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process training step if data available"""
        try:
            obs_vec = agent_data.get('obs_vec')
            reward = agent_data.get('reward')
            done = bool(agent_data.get('done', False))

            if obs_vec is not None and reward is not None:
                await self._record_training_step(obs_vec, float(reward), agent_data, done)

                if done:
                    await self._end_training_episode()

                return {
                    'training_step_processed': True,
                    'reward': float(reward),
                    'episode_done': bool(done),
                    'buffer_size': len(self.buffer['rewards'])
                }

            return {
                'training_step_processed': False,
                'reason': 'insufficient_data'
            }

        except Exception as e:
            self.logger.error(f"Training step processing failed: {e}")
            return {'training_step_processed': False, 'error': str(e)}

    async def _record_training_step(self, obs_vec: np.ndarray, reward: float,
                                    agent_data: Dict[str, Any], done: bool = False):
        """Record training step with comprehensive market data"""
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0

            context = agent_data['context']

            # Extract position information from context
            position = float(context.get('position', 0.0) or 0.0)
            unrealized_pnl = float(context.get('unrealized_pnl', 0.0) or 0.0)

            # Get lag features
            lag_features = self.get_lag_features()

            # Build extended observation
            if bool(self.genome["position_aware"]):
                position_features = np.array([
                    position,                                               # Current position
                    unrealized_pnl,                                        # Unrealized PnL
                    position * float(context.get('volatility', 1.0)),      # Risk-adjusted position
                    np.sign(position) * min(abs(position), 1.0),           # Position direction
                    abs(position) / max(abs(position), 1.0),               # Position size ratio (0..1)
                    float(len(self.position_history)) / 1000.0             # Experience level (0..1)
                ], dtype=np.float32)

                extended_obs = np.concatenate([obs_vec, lag_features, position_features])
            else:
                extended_obs = np.concatenate([obs_vec, lag_features])

            # Validate extended observation
            if np.any(np.isnan(extended_obs)):
                extended_obs = np.nan_to_num(extended_obs)

            # Pad or truncate to expected size
            expected = int(self.network.extended_obs_size)
            if len(extended_obs) < expected:
                padding = np.zeros(expected - len(extended_obs), dtype=np.float32)
                extended_obs = np.concatenate([extended_obs, padding])
            elif len(extended_obs) > expected:
                extended_obs = extended_obs[:expected]

            # Convert to tensors
            obs_tensor = torch.as_tensor(extended_obs, dtype=torch.float32, device=self.device)
            market_tensor = torch.as_tensor(lag_features, dtype=torch.float32, device=self.device)

            # Get network outputs
            with torch.no_grad():
                action_mean, action_std, value = self.network(obs_tensor.unsqueeze(0), market_tensor.unsqueeze(0))

                # Apply market adaptations
                if bool(self.genome["vol_scaling"]) and float(context.get('volatility', 1.0)) > 0:
                    vol_adjustment = float(np.sqrt(1.0 / max(float(context.get('volatility', 1.0)), 1e-9)))
                    action_std = action_std * vol_adjustment

                # Apply position-aware action scaling
                if bool(self.genome["position_aware"]) and abs(position) > 0.8:
                    position_penalty = 1.0 - min(abs(position), 1.0) * 0.2
                    action_std = action_std * position_penalty

                # Clamp action std
                action_std = torch.clamp(action_std, 0.01, 0.5)

                # Sample action
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

                # Validate outputs
                if torch.any(torch.isnan(action)):
                    action = torch.zeros_like(action)
                if torch.any(torch.isnan(log_prob)):
                    log_prob = torch.zeros_like(log_prob)
                if torch.any(torch.isnan(value)):
                    value = torch.zeros_like(value)

            # Store in buffer
            self.buffer['observations'].append(obs_tensor)
            self.buffer['market_features'].append(market_tensor)
            self.buffer['actions'].append(action.squeeze(0))
            self.buffer['log_probs'].append(log_prob.squeeze(0))
            self.buffer['values'].append(value.squeeze(0))
            self.buffer['rewards'].append(torch.tensor(float(reward), dtype=torch.float32, device=self.device))
            self.buffer['dones'].append(torch.tensor(bool(done), dtype=torch.bool, device=self.device))

            # Update position tracking
            self.position = position
            self.unrealized_pnl = unrealized_pnl
            self.position_history.append({
                'position': position,
                'unrealized_pnl': unrealized_pnl,
                'timestamp': datetime.now(),
                'volatility': float(context.get('volatility', 1.0))
            })

            # Update risk metrics
            self._update_risk_metrics()

            # Update trading metrics
            self._update_trading_metrics({'pnl': reward})

        except Exception as e:
            self.logger.error(f"Training step recording failed: {e}")

    async def _end_training_episode(self):
        """End training episode with PPO updates"""
        try:
            if len(self.buffer['rewards']) < int(self.cfg.min_episode_length):
                self.logger.warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping")
                self._clear_buffer()
                return

            # Calculate GAE advantages and returns
            await self._compute_gae_advantages()

            # Perform PPO updates
            update_stats = await self._perform_ppo_updates()

            # Update training statistics
            self._update_training_statistics(update_stats)

            # Track episode completion
            episode_reward = float(sum(r.item() for r in self.buffer['rewards']))
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(len(self.buffer['rewards']))
            self.training_stats['episodes_completed'] += 1

            # Clear buffer
            self._clear_buffer()

        except Exception as e:
            self.logger.error(f"Episode ending failed: {e}")
            self._clear_buffer()

    async def _compute_gae_advantages(self, final_value: float = 0.0):
        """Enhanced GAE computation with validation"""
        try:
            rewards = torch.stack(self.buffer['rewards'])
            values = torch.stack(self.buffer['values'])
            dones = torch.stack(self.buffer['dones'])

            # Validate tensors
            if torch.any(torch.isnan(rewards)):
                rewards = torch.nan_to_num(rewards)
            if torch.any(torch.isnan(values)):
                values = torch.nan_to_num(values)

            # GAE computation
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)

            next_value = float(final_value)
            next_advantage = 0.0

            for t in reversed(range(len(rewards) - 1)):
                if bool(dones[t].item()):
                    next_value = 0.0
                    next_advantage = 0.0

                td_error = rewards[t] + float(self.cfg.gamma) * next_value - values[t]
                advantages[t] = td_error + float(self.cfg.gamma) * float(self.cfg.gae_lambda) * next_advantage
                returns[t] = rewards[t] + float(self.cfg.gamma) * next_value

                next_value = float(values[t].item())
                next_advantage = float(advantages[t].item())

            # Adaptive advantage normalization
            std_val = float(advantages.std().item()) if len(advantages) > 0 else 0.0
            if std_val > 1e-6:
                self.running_adv_std = float(self.genome["adv_decay"]) * self.running_adv_std + (1 - float(self.genome["adv_decay"])) * std_val
                advantages = advantages / (self.running_adv_std + 1e-8)

            # Store in buffer
            self.buffer['advantages'] = [advantages[i] for i in range(len(advantages))]
            self.buffer['returns'] = [returns[i] for i in range(len(returns))]

        except Exception as e:
            self.logger.error(f"GAE computation failed: {e}")
            await self._fallback_advantage_computation()

    async def _fallback_advantage_computation(self, final_value: float = 0.0):
        """Fallback advantage computation"""
        try:
            returns: List[float] = []
            running_return = float(final_value)
            for reward in reversed(self.buffer['rewards']):
                running_return = float(reward.item()) + float(self.cfg.gamma) * running_return
                returns.insert(0, running_return)

            values = [float(v.item()) for v in self.buffer['values']]
            advantages = [ret - val for ret, val in zip(returns, values)]

            # Normalize
            if len(advantages) > 1:
                adv_mean = float(np.mean(advantages))
                adv_std = float(np.std(advantages))
                if adv_std > 1e-6:
                    advantages = [(a - adv_mean) / adv_std for a in advantages]

            self.buffer['returns'] = [torch.tensor(r, device=self.device) for r in returns]
            self.buffer['advantages'] = [torch.tensor(a, device=self.device) for a in advantages]

        except Exception as e:
            self.logger.error(f"Fallback advantage computation failed: {e}")

    async def _perform_ppo_updates(self) -> Dict[str, float]:
        """Advanced PPO updates with market awareness"""
        try:
            # Prepare data
            observations = torch.stack(self.buffer['observations'])
            market_features = torch.stack(self.buffer['market_features'])
            actions = torch.stack(self.buffer['actions'])
            old_log_probs = torch.stack(self.buffer['log_probs'])
            returns = torch.stack(self.buffer['returns'])
            advantages = torch.stack(self.buffer['advantages'])

            # Sanitize data
            observations = torch.nan_to_num(observations)
            market_features = torch.nan_to_num(market_features)
            actions = torch.nan_to_num(actions)
            old_log_probs = torch.nan_to_num(old_log_probs)
            returns = torch.nan_to_num(returns)
            advantages = torch.nan_to_num(advantages)

            total_actor_loss = 0.0
            total_critic_loss = 0.0
            total_entropy = 0.0
            total_kl_div = 0.0
            update_count = 0

            # Multiple epochs
            for _ in range(int(self.cfg.ppo_epochs)):
                indices = torch.randperm(len(observations), device=self.device)

                for start in range(0, len(observations), int(self.cfg.batch_size)):
                    end = start + int(self.cfg.batch_size)
                    batch_indices = indices[start:end]

                    batch_obs = observations[batch_indices]
                    batch_market = market_features[batch_indices]
                    batch_actions = actions[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]

                    # Forward pass
                    action_mean, action_std, values = self.network(batch_obs, batch_market)

                    # Create distribution
                    dist = torch.distributions.Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)

                    # KL divergence for early stopping (approx)
                    kl_div = (batch_old_log_probs - new_log_probs).mean()

                    if float(kl_div.item()) > float(self.cfg.target_kl) * 2.0:
                        break  # early stop this epoch

                    # Policy loss
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    ratio = torch.clamp(ratio, 0.1, 10.0)  # Stability clamp

                    surr1 = ratio * batch_advantages
                    surr2 = torch.clamp(ratio, 1 - self.adaptive_clip_eps, 1 + self.adaptive_clip_eps) * batch_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    value_pred = values.squeeze(-1)
                    value_loss = F.mse_loss(value_pred, batch_returns)

                    # Entropy loss
                    entropy_loss = -entropy.mean()

                    # Total losses
                    total_actor_loss_batch = actor_loss + float(self.cfg.entropy_coeff) * entropy_loss
                    total_critic_loss_batch = float(self.cfg.value_coeff) * value_loss

                    if torch.isnan(total_actor_loss_batch) or torch.isnan(total_critic_loss_batch):
                        continue

                    # Actor update
                    self.actor_optimizer.zero_grad()
                    total_actor_loss_batch.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.network.feature_extractor.parameters()) +
                        list(self.network.actor_attention.parameters()) +
                        list(self.network.actor_head.parameters()) +
                        [self.network.log_std_base] +
                        list(self.network.std_conditioner.parameters()),
                        float(self.cfg.max_grad_norm)
                    )
                    self.actor_optimizer.step()

                    # Critic update
                    self.critic_optimizer.zero_grad()
                    total_critic_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.network.market_encoder.parameters()) +
                        list(self.network.value_head_1.parameters()) +
                        list(self.network.value_head_2.parameters()),
                        float(self.cfg.max_grad_norm)
                    )
                    self.critic_optimizer.step()

                    # Accumulate statistics
                    total_actor_loss += float(actor_loss.item())
                    total_critic_loss += float(value_loss.item())
                    total_entropy += float(entropy.mean().item())
                    total_kl_div += float(kl_div.item())
                    update_count += 1

            if update_count > 0:
                update_stats = {
                    'actor_loss': total_actor_loss / update_count,
                    'critic_loss': total_critic_loss / update_count,
                    'entropy': total_entropy / update_count,
                    'kl_divergence': total_kl_div / update_count,
                    'updates_performed': float(update_count)
                }
            else:
                update_stats = {
                    'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0,
                    'kl_divergence': 0.0, 'updates_performed': 0.0
                }

            return update_stats

        except Exception as e:
            self.logger.error(f"PPO update failed: {e}")
            return {'actor_loss': 0.0, 'critic_loss': 0.0, 'entropy': 0.0, 'kl_divergence': 0.0, 'updates_performed': 0.0}

    def _update_training_statistics(self, update_stats: Dict[str, float]):
        """Update comprehensive training statistics"""
        self.training_stats['total_updates'] += 1

        # Update loss trends
        if self.training_stats['total_updates'] > 1:
            self.training_stats['actor_loss_trend'] = (
                0.9 * float(self.training_stats['actor_loss_trend']) +
                0.1 * float(update_stats['actor_loss'])
            )
            self.training_stats['critic_loss_trend'] = (
                0.9 * float(self.training_stats['critic_loss_trend']) +
                0.1 * float(update_stats['critic_loss'])
            )
        else:
            self.training_stats['actor_loss_trend'] = float(update_stats['actor_loss'])
            self.training_stats['critic_loss_trend'] = float(update_stats['critic_loss'])

        # Update other metrics
        self.training_stats['kl_divergence'] = float(update_stats['kl_divergence'])
        self.training_stats['policy_entropy'] = float(update_stats['entropy'])

        # Explained variance (safe)
        if len(self.buffer['returns']) > 0 and len(self.buffer['values']) > 0:
            returns_np = np.array([float(r.item()) for r in self.buffer['returns']])
            values_np = np.array([float(v.item()) for v in self.buffer['values']])
            var_r = float(np.var(returns_np))
            if var_r > 1e-6:
                explained_var = 1.0 - float(np.var(returns_np - values_np)) / var_r
                self.training_stats['explained_variance'] = max(0.0, float(explained_var))

    # ─────────────────────────────────────────────────────────
    # Metrics & health
    # ─────────────────────────────────────────────────────────
    def _update_risk_metrics(self):
        """Update position and risk metrics"""
        if len(self.position_history) >= 10:
            recent = list(self.position_history)[-50:]
            recent_positions = [float(p['position']) for p in recent]
            recent_pnls = [float(p['unrealized_pnl']) for p in recent]

            self.risk_metrics['max_position'] = max(abs(p) for p in recent_positions)
            self.risk_metrics['avg_position'] = float(np.mean(np.abs(recent_positions)))
            self.risk_metrics['position_volatility'] = float(np.std(recent_positions))

            if self.risk_metrics['position_volatility'] > 0:
                avg_return = float(np.mean(recent_pnls))
                self.risk_metrics['risk_adjusted_return'] = float(avg_return / self.risk_metrics['position_volatility'])
            else:
                self.risk_metrics['risk_adjusted_return'] = 0.0

    async def _update_performance_tracking(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update performance tracking metrics"""
        try:
            if len(self.episode_rewards) > 0:
                latest_reward = float(self.episode_rewards[-1])
                context = agent_data['context']

                # Track performance by regime
                regime = context.get('regime', 'unknown')
                self.market_performance[regime]['rewards'].append(latest_reward)
                self.market_performance[regime]['count'] += 1

                # Track performance by volatility
                vol_level = context.get('volatility_level', 'medium')
                self.volatility_performance[vol_level]['rewards'].append(latest_reward)
                self.volatility_performance[vol_level]['count'] += 1

            return {
                'performance_tracked': True,
                'total_episodes': len(self.episode_rewards),
                'avg_reward': float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0,
                'market_regimes_tracked': len(self.market_performance),
                'volatility_levels_tracked': len(self.volatility_performance)
            }

        except Exception as e:
            self.logger.error(f"Performance tracking update failed: {e}")
            return {'performance_tracked': False, 'error': str(e)}

    def _clear_buffer(self):
        """Clear experience buffer"""
        for key in self.buffer:
            self.buffer[key].clear()

    async def _generate_agent_thesis(self, agent_data: Dict[str, Any],
                                     agent_result: Dict[str, Any]) -> str:
        """Generate comprehensive agent thesis"""
        try:
            episodes_completed = int(self.training_stats['episodes_completed'])
            total_updates = int(self.training_stats['total_updates'])
            network_params = int(sum(p.numel() for p in self.network.parameters()))

            avg_reward = float(np.mean(list(self.episode_rewards)[-10:])) if self.episode_rewards else 0.0
            market_adapted = bool(agent_result.get('market_adapted', False))

            thesis_parts = [
                f"PPO-Lag Agent: {episodes_completed} episodes, {total_updates} updates with {network_params:,} parameters",
                f"Performance: {avg_reward:.3f} avg reward with market adaptation {'active' if market_adapted else 'inactive'}"
            ]

            if agent_result.get('training_step_processed', False):
                reward = float(agent_result.get('reward', 0.0) or 0.0)
                buffer_size = int(agent_result.get('buffer_size', 0) or 0)
                thesis_parts.append(f"Training: {reward:.3f} reward step, {buffer_size} buffer entries")

            if market_adapted:
                regime = agent_result.get('regime', 'unknown')
                adaptive_clip = float(agent_result.get('adaptive_clip', self.adaptive_clip_eps))
                thesis_parts.append(f"Market adaptation: {regime} regime with {adaptive_clip:.3f} clip epsilon")

            thesis_parts.append(f"Position tracking: {self.position:.3f} current, {len(self.position_history)} history entries")

            max_position = float(self.risk_metrics['max_position'])
            risk_adj_return = float(self.risk_metrics['risk_adjusted_return'])
            thesis_parts.append(f"Risk metrics: {max_position:.3f} max position, {risk_adj_return:.3f} risk-adjusted return")

            return " | ".join(thesis_parts)

        except Exception as e:
            return f"Agent thesis generation failed: {str(e)} - PPO-Lag training continuing"

    async def _update_agent_smart_bus(self, agent_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with PPO-Lag agent results (namespaced metrics to avoid collisions)."""
        try:
            # Agent status
            agent_status = {
                'agent_type': 'ppo-lag',
                'episodes_completed': self.training_stats['episodes_completed'],
                'total_updates': self.training_stats['total_updates'],
                'network_parameters': int(sum(p.numel() for p in self.network.parameters())),
                'training_active': len(self.buffer['rewards']) > 0,
                'buffer_size': len(self.buffer['rewards'])
            }
            self.smart_bus.set('agent_status', agent_status, module='PPOLagAgent', thesis=thesis)

            # Namespaced training metrics
            training_metrics = {
                'training_stats': dict(self.training_stats),
                'episode_rewards': list(self.episode_rewards)[-10:],
                'episode_lengths': list(self.episode_lengths)[-10:],
                'adaptive_parameters': {
                    'clip_eps': self.adaptive_clip_eps,
                    'lr_factor': self.adaptive_lr_factor,
                    'entropy_coeff': self.cfg.entropy_coeff
                }
            }
            self.smart_bus.set(
                'ppo_lag_training_metrics',
                training_metrics,
                module='PPOLagAgent',
                thesis=f"PPO-Lag training metrics: {self.training_stats['episodes_completed']} episodes completed"
            )

            # Position metrics
            position_metrics = {
                'current_position': float(self.position),
                'unrealized_pnl': float(self.unrealized_pnl),
                'risk_metrics': dict(self.risk_metrics),
                'position_history_size': len(self.position_history)
            }
            self.smart_bus.set('position_metrics', position_metrics, module='PPOLagAgent', thesis="Position tracking and risk metrics")

            # Market adaptation
            market_adaptation = {
                'lag_window': int(self.genome["lag_window"]),
                'buffer_sizes': {
                    'price': len(self.price_buffer),
                    'volume': len(self.volume_buffer),
                    'spread': len(self.spread_buffer),
                    'volatility': len(self.volatility_buffer)
                },
                'market_performance': {
                    regime: {
                        'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                        'count': int(data['count'])
                    } for regime, data in self.market_performance.items()
                },
                'volatility_performance': {
                    vol_level: {
                        'avg_reward': float(np.mean(list(data['rewards'])[-10:])) if len(data['rewards']) >= 10 else 0.0,
                        'count': int(data['count'])
                    } for vol_level, data in self.volatility_performance.items()
                },
                'adaptation_settings': {
                    'vol_scaling': bool(self.genome["vol_scaling"]),
                    'position_aware': bool(self.genome["position_aware"])
                }
            }
            self.smart_bus.set('market_adaptation', market_adaptation, module='PPOLagAgent', thesis="Market adaptation and lag feature processing")

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ─────────────────────────────────────────────────────────
    # Errors & fallbacks
    # ─────────────────────────────────────────────────────────
    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no agent data is available"""
        self.logger.warning("No agent data available - using cached state")

        return {
            'agent_type': 'ppo-lag',
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': int(sum(p.numel() for p in self.network.parameters())),
            'fallback_reason': 'no_agent_data'
        }

    async def _handle_agent_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle agent operation errors"""
        processing_time = (time.time() - start_time) * 1000.0

        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()

        if self.circuit_breaker['failures'] >= int(self.circuit_breaker['threshold']):
            self.circuit_breaker['state'] = 'OPEN'

        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "PPOLagAgent")  # noqa: F841 (context used by pinpointer)
        explanation = self.english_explainer.explain_error(
            "PPOLagAgent", str(error), "agent operations"
        )

        self.logger.error(
            format_operator_message(
                "[CRASH]", "AGENT_OPERATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="ppo_lag_training"
            )
        )

        # Record failure
        self._record_failure(error)

        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'agent_type': 'ppo-lag',
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': int(sum(p.numel() for p in self.network.parameters())),
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    # ─────────────────────────────────────────────────────────
    # Health & analytics
    # ─────────────────────────────────────────────────────────
    def _update_agent_health(self):
        """Update agent health metrics"""
        try:
            if self.training_stats['episodes_completed'] > 10:
                recent_rewards = list(self.episode_rewards)[-10:]
                if recent_rewards:
                    avg_reward = float(np.mean(recent_rewards))
                    if avg_reward < -20:
                        self._health_status = 'warning'
                    elif avg_reward > 10:
                        self._health_status = 'healthy'
                    else:
                        self._health_status = 'healthy'

            # Check for NaN in network parameters
            has_nan = any(torch.isnan(p).any().item() for p in self.network.parameters() if p.requires_grad and p.data.dtype.is_floating_point)
            if has_nan:
                self._health_status = 'critical'

            self._last_health_check = time.time()

        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_agent_performance(self):
        """Analyze agent performance metrics"""
        try:
            if len(self.episode_rewards) > 20:
                recent_performance = float(np.mean(list(self.episode_rewards)[-10:]))
                overall_performance = float(np.mean(list(self.episode_rewards)))
                if overall_performance == 0:
                    return
                if recent_performance > overall_performance * 1.2:
                    self.logger.info(
                        format_operator_message(
                            "[CHART]", "PERFORMANCE_IMPROVEMENT",
                            recent_avg=f"{recent_performance:.3f}",
                            overall_avg=f"{overall_performance:.3f}",
                            episodes=len(self.episode_rewards),
                            context="agent_performance"
                        )
                    )

        except Exception as e:
            self.logger.error(f"Performance analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'PPOLagAgent', 'agent_cycle', float(processing_time), True
        )

        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'PPOLagAgent', 'agent_cycle', 0.0, False
        )

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in agent recommendations"""
        try:
            base_confidence = 0.5

            # Confidence from training progress
            if self.training_stats['episodes_completed'] > 10:
                recent_rewards = list(self.episode_rewards)[-10:]
                if recent_rewards:
                    avg_reward = float(np.mean(recent_rewards))
                    reward_std = float(np.std(recent_rewards))

                    if avg_reward > 5:
                        base_confidence += 0.3
                    elif avg_reward < -5:
                        base_confidence -= 0.2

                    if reward_std < 10:
                        base_confidence += 0.2

            # Model stability
            explained_variance = float(self.training_stats['explained_variance'])
            base_confidence += explained_variance * 0.2

            # Position management
            if float(self.risk_metrics['risk_adjusted_return']) > 0:
                base_confidence += 0.1

            # Action-specific adjustments
            if isinstance(action, dict):
                action_magnitude = float(action.get('magnitude', 0.5))
                if action_magnitude > 0.8:
                    base_confidence += 0.1
                elif action_magnitude < 0.3:
                    base_confidence -= 0.1

            return float(np.clip(base_confidence, 0.1, 1.0))

        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    # ─────────────────────────────────────────────────────────
    # Persistence & legacy
    # ─────────────────────────────────────────────────────────
    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'config': asdict(self.cfg),
            'genome': dict(self.genome),
            'training_stats': dict(self.training_stats),
            'position': float(self.position),
            'unrealized_pnl': float(self.unrealized_pnl),
            'risk_metrics': dict(self.risk_metrics),
            'episode_rewards': list(self.episode_rewards),
            'episode_lengths': list(self.episode_lengths),
            'market_performance': {k: {'rewards': list(v['rewards']), 'count': int(v['count'])} for k, v in self.market_performance.items()},
            'volatility_performance': {k: {'rewards': list(v['rewards']), 'count': int(v['count'])} for k, v in self.volatility_performance.items()},
            'position_history': list(self.position_history)[-100:],  # Keep recent history
            'adaptive_parameters': {
                'running_adv_std': float(self.running_adv_std),
                'adaptive_clip_eps': float(self.adaptive_clip_eps),
                'adaptive_lr_factor': float(self.adaptive_lr_factor)
            },
            'network_state': self.network.state_dict(),
            'actor_optimizer_state': self.actor_optimizer.state_dict(),
            'critic_optimizer_state': self.critic_optimizer.state_dict(),
            'health_status': self._health_status,
            'circuit_breaker': dict(self.circuit_breaker)
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'config' in state and isinstance(state['config'], dict):
            self._update_cfg_from_dict(state['config'])

        if 'genome' in state and isinstance(state['genome'], dict):
            self.genome.update(state['genome'])

        if 'training_stats' in state and isinstance(state['training_stats'], dict):
            self.training_stats.update(state['training_stats'])

        if 'position' in state:
            self.position = float(state['position'])

        if 'unrealized_pnl' in state:
            self.unrealized_pnl = float(state['unrealized_pnl'])

        if 'risk_metrics' in state and isinstance(state['risk_metrics'], dict):
            self.risk_metrics.update({k: float(v) for k, v in state['risk_metrics'].items()})

        if 'episode_rewards' in state:
            self.episode_rewards = deque([float(x) for x in state['episode_rewards']], maxlen=200)

        if 'episode_lengths' in state:
            self.episode_lengths = deque([int(x) for x in state['episode_lengths']], maxlen=200)

        if 'market_performance' in state and isinstance(state['market_performance'], dict):
            for k, v in state['market_performance'].items():
                self.market_performance[k]['rewards'] = deque([float(x) for x in v.get('rewards', [])], maxlen=100)
                self.market_performance[k]['count'] = int(v.get('count', 0))

        if 'volatility_performance' in state and isinstance(state['volatility_performance'], dict):
            for k, v in state['volatility_performance'].items():
                self.volatility_performance[k]['rewards'] = deque([float(x) for x in v.get('rewards', [])], maxlen=100)
                self.volatility_performance[k]['count'] = int(v.get('count', 0))

        if 'position_history' in state:
            self.position_history = deque(state['position_history'], maxlen=1000)

        if 'adaptive_parameters' in state and isinstance(state['adaptive_parameters'], dict):
            params = state['adaptive_parameters']
            self.running_adv_std = float(params.get('running_adv_std', 1.0))
            self.adaptive_clip_eps = float(params.get('adaptive_clip_eps', self.cfg.clip_eps))
            self.adaptive_lr_factor = float(params.get('adaptive_lr_factor', 1.0))

        if 'network_state' in state:
            self.network.load_state_dict(state['network_state'])

        if 'actor_optimizer_state' in state:
            self.actor_optimizer.load_state_dict(state['actor_optimizer_state'])

        if 'critic_optimizer_state' in state:
            self.critic_optimizer.load_state_dict(state['critic_optimizer_state'])

        if 'health_status' in state:
            self._health_status = state['health_status']

        if 'circuit_breaker' in state and isinstance(state['circuit_breaker'], dict):
            self.circuit_breaker.update(state['circuit_breaker'])

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'episodes_completed': self.training_stats['episodes_completed'],
            'network_parameters': int(sum(p.numel() for p in self.network.parameters())),
            'buffer_size': len(self.buffer['rewards'])
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def record_step(self, obs_vec: np.ndarray, reward: float, **market_data):
        """Legacy compatibility for step recording"""
        try:
            if np.any(np.isnan(obs_vec)):
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                reward = 0.0

            agent_data = {
                'obs_vec': obs_vec,
                'reward': float(reward),
                'context': {
                    'price': float(market_data.get('price', 0.0) or 0.0),
                    'volume': float(market_data.get('volume', 0.0) or 0.0),
                    'spread': float(market_data.get('spread', 0.0) or 0.0),
                    'volatility': float(market_data.get('volatility', 1.0) or 1.0),
                    'position': float(market_data.get('position', 0.0) or 0.0),
                    'unrealized_pnl': float(market_data.get('unrealized_pnl', 0.0) or 0.0)
                }
            }

            asyncio.create_task(self._record_training_step(obs_vec, float(reward), agent_data))

        except Exception as e:
            self.logger.error(f"Step recording failed: {e}")

    def end_episode(self, *args, **kwargs):
        """Legacy compatibility for episode ending"""
        try:
            asyncio.create_task(self._end_training_episode())
        except Exception as e:
            self.logger.error(f"Episode ending failed: {e}")

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Enhanced action selection with market awareness"""
        try:
            if torch.any(torch.isnan(obs_tensor)):
                obs_tensor = torch.nan_to_num(obs_tensor)

            # Pad/trim observation to expected size
            expected = int(self.network.extended_obs_size)
            if obs_tensor.shape[-1] < expected:
                pad = torch.zeros((*obs_tensor.shape[:-1], expected - obs_tensor.shape[-1]), device=obs_tensor.device, dtype=obs_tensor.dtype)
                obs_tensor = torch.cat([obs_tensor, pad], dim=-1)
            elif obs_tensor.shape[-1] > expected:
                obs_tensor = obs_tensor[..., :expected]

            lag_features_size = int(self.genome["lag_window"]) * 4
            market_features = obs_tensor[..., -lag_features_size:]

            with torch.no_grad():
                action_mean, action_std, _ = self.network(obs_tensor, market_features)
                dist = torch.distributions.Normal(action_mean, action_std)
                action = dist.sample()
                if torch.any(torch.isnan(action)):
                    action = torch.zeros_like(action)
                action = torch.clamp(action, -2.0, 2.0)
                return action

        except Exception as e:
            self.logger.error(f"Action selection failed: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, int(self.genome["act_size"]), device=self.device)

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with market and position awareness"""
        try:
            position_norm = float(np.tanh(self.position))
            unrealized_pnl_norm = float(np.tanh(self.unrealized_pnl / 100.0))

            max_pos_norm = float(np.tanh(self.risk_metrics['max_position']))
            risk_adj_return = float(np.tanh(self.risk_metrics['risk_adjusted_return']))

            if len(self.episode_rewards) > 0:
                avg_reward = float(np.mean(list(self.episode_rewards)[-10:]))
                reward_trend = 0.0
                if len(self.episode_rewards) >= 5:
                    recent_rewards = list(self.episode_rewards)[-5:]
                    reward_trend = float(np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0])
            else:
                avg_reward = 0.0
                reward_trend = 0.0

            episodes_norm = min(1.0, float(self.training_stats['episodes_completed']) / 1000.0)
            kl_div_norm = float(np.tanh(float(self.training_stats['kl_divergence']) * 100.0))
            entropy_norm = float(np.tanh(float(self.training_stats['policy_entropy']) * 10.0))

            adaptive_clip_norm = float(self.adaptive_clip_eps) / 0.3
            adaptive_lr_norm = float(self.adaptive_lr_factor)

            buffer_fullness = len(self.buffer['rewards']) / 200.0
            experience_level = min(1.0, len(self.position_history) / 1000.0)

            observation = np.array([
                position_norm,
                unrealized_pnl_norm,
                max_pos_norm,
                risk_adj_return,
                avg_reward / 100.0,
                reward_trend / 10.0,
                episodes_norm,
                kl_div_norm,
                entropy_norm,
                adaptive_clip_norm,
                adaptive_lr_norm,
                buffer_fullness,
                experience_level,
                float(self.training_stats['explained_variance'])
            ], dtype=np.float32)

            if np.any(np.isnan(observation)):
                observation = np.nan_to_num(observation)

            observation = np.clip(observation, -3.0, 3.0)
            return observation

        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(14, dtype=np.float32)

    def step(self, *args, **kwargs):
        """Legacy step method"""
        pass

    def get_weights(self) -> Dict[str, Any]:
        """Get network weights"""
        return {'network': self.network.state_dict()}

    def get_gradients(self) -> Dict[str, Any]:
        """Get current gradients"""
        gradients: Dict[str, Optional[np.ndarray]] = {}
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.detach().cpu().numpy()
            else:
                gradients[name] = None
        return gradients

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose action based on current agent state"""
        try:
            obs_vec = inputs.get('obs_vec')
            if obs_vec is not None:
                obs_arr = np.array(obs_vec, dtype=np.float32)
                obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
                lag_features = torch.as_tensor(self.get_lag_features(), dtype=torch.float32, device=self.device).unsqueeze(0)

                with torch.no_grad():
                    action_mean, action_std, value = self.network(obs_tensor, lag_features)
                    action = torch.normal(action_mean, action_std)
                    action = torch.clamp(action, -1.0, 1.0)

                confidence = float(torch.mean(1.0 / (1.0 + action_std)).item())
                return {
                    'action_type': 'trading_signal',
                    'action_values': action.cpu().numpy().flatten().tolist(),
                    'confidence': confidence,
                    'value_estimate': float(value.item()),
                    'reasoning': f"PPO-Lag agent action based on {len(self.price_buffer)} market observations"
                }
            else:
                return {
                    'action_type': 'no_action',
                    'action_values': [0.0] * int(self.genome["act_size"]),
                    'confidence': 0.1,
                    'value_estimate': 0.0,
                    'reasoning': 'No observation data available for action selection'
                }

        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'error',
                'action_values': [0.0] * int(self.genome["act_size"]),
                'confidence': 0.0,
                'value_estimate': 0.0,
                'reasoning': f'Action proposal error: {str(e)}'
            }

    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        base_confidence = 0.5
        if hasattr(self, 'training_stats') and self.training_stats['episodes_completed'] > 10:
            recent_rewards = list(self.episode_rewards)[-10:] if hasattr(self, 'episode_rewards') else []
            if recent_rewards:
                avg_reward = float(np.mean(recent_rewards))
                if avg_reward > 5:
                    base_confidence += 0.3
                elif avg_reward < -5:
                    base_confidence -= 0.2
        return float(np.clip(base_confidence, 0.1, 1.0))
