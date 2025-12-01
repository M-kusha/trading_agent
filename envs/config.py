# envs/config.py
"""
Enhanced Configuration System for InfoBus-Integrated Trading Environment
BUS-FIRST edition: modules are source-of-truth; config is fallback + guard-rails.

Key changes:
- Added bus-first policy toggles (prefer_bus_* etc.) to avoid duplicating module logic
- Added missing fields referenced by ModernTradingEnv (environment_observation_size, prefer_bus_data, fees, etc.)
- Kept all prior fields for backward compatibility (act as defaults if bus has no data)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional
from pathlib import Path


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _ensure_dir(p: str) -> None:
    if p:
        Path(p).mkdir(parents=True, exist_ok=True)


def _to_builtin(obj: Any) -> Any:
    """Recursively convert values to JSON-serializable builtins."""
    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (list, tuple)):
            return [_to_builtin(x) for x in obj]
        if isinstance(obj, dict):
            return {str(k): _to_builtin(v) for k, v in obj.items()}
    except Exception:
        pass
    return str(obj)


# ─────────────────────────────────────────────────────────
# TradingConfig (bus-first; config = fallback + guard rails)
# ─────────────────────────────────────────────────────────
@dataclass
class TradingConfig:
    """Centralized configuration for InfoBus-integrated trading environment (BUS-FIRST)."""

    # ===================================================================
    # Core Environment Parameters (fallbacks)
    # ===================================================================
    initial_balance: float = 3000.0
    max_steps: int = 100000  # increase default episode length to reduce frequent resets
    debug: bool = True
    init_seed: int = 42
    max_steps_per_episode: int = field(init=False)

    # Observation sizing (env reads this, but will prefer bus features if available)
    environment_observation_size: int = 256

    # Primary timeframe (used mainly for local fallback data windows)
    primary_timeframe: str = "H1"

    # Minimum data bars required (prevents 4-step episodes from tiny datasets)
    # Set to 50 to ensure at least 50 timesteps per episode
    min_required_data_bars: int = 50

    # ===================================================================
    # InfoBus & Orchestrator
    # ===================================================================
    info_bus_enabled: bool = True
    info_bus_audit_level: str = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
    info_bus_validation: bool = True
    info_bus_init_timeout: float = 2.0
    orchestrator_init_timeout: float = 10.0
    orchestrator_async_init: bool = True
    # When the env triggers the orchestrator each step, optionally wait a few ms
    # so decision modules (e.g., PositionManager) can enqueue orders before the
    # env collects intents. Keeps bus-first async behavior, but reduces empty cycles.
    orchestrator_sync_wait_ms: float = 25.0
    # Limit the number of concurrent orchestrator executions scheduled by the env
    orchestrator_max_inflight: int = 1
    # Only schedule orchestrator once every N env steps (1 = every step)
    orchestrator_step_interval: int = 1

    # Optional general step throttle (ms) applied even when orchestrator is disabled
    # or when no orchestrator scheduling occurs on a given step. Default 0 (no delay).
    step_sleep_ms: float = 0.0

    # Bus-first policy toggles (single source of truth = modules via SmartInfoBus)
    bus_first: bool = True
    prefer_bus_data: bool = True          # prefer MarketDataProvider over local data
    prefer_bus_features: bool = True      # prefer AdvancedFeatureEngine/MultiScaleFeatureEngine
    prefer_bus_rewards: bool = True       # prefer RiskAdjustedReward.shaped_reward
    prefer_bus_metrics: bool = True       # prefer PortfolioRiskSystem/DrawdownRescue/etc
    prefer_bus_limits: bool = True        # prefer Compliance/PortfolioRiskSystem/etc
    allow_module_overrides: bool = True   # let modules override defaults at runtime
    halt_on_emergency: bool = True        # stop new orders on emergency/risk kill switch

    bus_aliases: Dict[str, List[str]] = field(default_factory=lambda: {
        # Data / step
        "market_data": ["MarketDataProvider"],
        "multi_timeframe_data": ["MarketDataProvider"],
        "step_idx": ["MarketDataProvider"],

        # Positions / portfolio state  (single writer = Executor)
        "positions": ["Executor"],
        "equity": ["Executor"],
        "balance": ["Executor"],
        "trades": ["Executor"],
        "portfolio_metrics": ["Executor"],
        "trading_result": ["Executor"],

        # Order & execution rollups  (consumed by Compliance/EQM/etc.)
        "order_data": ["Executor"],
        "execution_data": ["Executor"],
        "execution_reports": ["Executor"],

        # Helpful aliases (kept canonical)
        "current_positions": ["Executor"],
        "pnl_data": ["Executor"],

        # Features / observations
        "advanced_features": ["AdvancedFeatureEngine", "MultiScaleFeatureEngine"],

        # Reward shaping
        "shaped_reward": ["RiskAdjustedReward"],

        # Risk metrics and limits
        "risk_metrics": ["PortfolioRiskSystem"],
        "risk_score": ["PortfolioRiskSystem"],
        "drawdown_risk": ["DrawdownRescue"],
        "correlation_risk": ["CorrelatedRiskController"],
        "risk_limits": ["ComplianceModule"],
        "position_limits": ["PortfolioRiskSystem"],
        "trade_limits": ["RoleCoach"],
        "mode_thresholds": ["TradingModeManager"],
        "risk_scaling_factor": ["UnifiedMarketModule"],   # was TimeAwareRiskScaling
        "risk_scaling": ["DynamicRiskController"],

        # Kill switches
        "emergency_mode": ["SessionManager"],
        "risk_alerts": ["DynamicRiskController"],

        # Regime/volatility/session (context)
        "market_regime": ["UnifiedMarketModule"],         # was FractalRegimeConfirmation
        "volatility_data": ["MarketDataProvider"],        # drop TimeAwareRiskScaling
        "trading_session": ["MarketDataProvider"],        # SessionManager doesn't provide this
    })


    # ===================================================================
    # Data and Instruments (used as defaults/fallbacks)
    # ===================================================================
    data_dir: str = "data/processed"
    instruments: List[str] = field(default_factory=lambda: ["EUR_USD", "XAU_USD"])
    timeframes: List[str] = field(default_factory=lambda: ["H1", "H4", "D1"])

    # ===================================================================
    # Trading Parameters (fallback only; modules own live values)
    # ===================================================================
    no_trade_penalty: float = 0.2          # Reduced - don't force trading
    consensus_min: float = 0.50            # Raised from 0.30 - need more agreement
    consensus_max: float = 0.85            # Raised from 0.70 - higher ceiling
    max_episodes: int = 10000

    # Execution economics (used by embedded executor when PositionManager/ExecutionQualityMonitor
    # do not provide an override; otherwise treated as fallback)
    default_spread: float = 0.0
    slippage_pts: float = 0.0
    commission_per_million: float = 0.0

    # Soft gating for env-embedded logic (fallback only; modules should own these live)
    min_confidence: float = 0.35           # Raised from 0.0 - require confidence
    min_intensity: float = 0.35            # Raised from 0.25 - filter weak signals
    ignore_hold: bool = True

    # ===================================================================
    # Risk Management (fallback guard-rails; Compliance/PortfolioRiskSystem are canonical)
    # ===================================================================
    rotation_gap: int = 5
    max_position_pct: float = 0.15       # Reduced from 0.25 - smaller positions = less risk
    max_total_exposure: float = 0.35     # Reduced from 0.50 - less total exposure
    max_drawdown: float = 0.15           # Reduced from 0.20 - tighter DD limit
    max_correlation: float = 0.7         # Reduced from 0.8 - less correlated risk

    # Position Management Specific (fallbacks)
    max_consecutive_losses: int = 5
    loss_reduction: float = 0.2
    max_instrument_concentration: float = 0.25
    min_volatility: float = 0.015
    hard_loss_eur: float = 30.0
    trail_pct: float = 0.10
    trail_abs_eur: float = 10.0
    pips_tolerance: int = 20
    min_size_pct: float = 0.01
    min_signal_threshold: float = 0.15
    position_scale_threshold: float = 0.30
    emergency_close_threshold: float = 0.85
    confidence_decay: float = 0.95

    # Emergency behavior tuning (centralized, used by PositionManager and others)
    emergency_drawdown_trigger: float = 0.15   # trigger if current drawdown > 15%
    emergency_risk_score_threshold: float = 0.7  # require risk_score >= 0.7 to escalate
    emergency_breach_steps: int = 2             # consecutive steps required before hard emergency action
    emergency_warmup_steps: int = 20            # ignore emergency checks for first N steps

    # Performance thresholds for Position Manager (fallbacks)
    position_max_processing_time_ms: float = 100
    position_circuit_breaker_threshold: int = 3

    # Enhanced Risk Parameters for InfoBus (scheduler defaults)
    risk_check_frequency: int = 1  # Steps between risk checks
    risk_alert_cooldown: int = 5   # Steps between similar alerts
    max_concurrent_alerts: int = 10

    # ===================================================================
    # Training Environment (env-level toggles; agent owns PPO hparams)
    # ===================================================================
    num_envs: int = 1
    test_mode: bool = False
    live_mode: bool = False
    training_mode: bool = False
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False

    # Module Enablement Flags (coarse)
    enable_meta_rl: bool = True
    enable_memory_systems: bool = True
    enable_strategy_evolution: bool = True
    enable_risk_monitoring: bool = True
    enable_visualization: bool = True

    # ===================================================================
    # PPO Hyperparameters (kept for legacy; agents (PPO/PPOLag) should own live HPs)
    # ===================================================================
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = 0.01

    # ===================================================================
    # Network Architecture (legacy defaults; agent modules own live nets)
    # ===================================================================
    policy_hidden_size: int = 256
    value_hidden_size: int = 256

    # ===================================================================
    # Training Schedule (env/trainer level schedules)
    # ===================================================================
    final_training_steps: int = 100000
    log_interval: int = 10
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    n_eval_episodes: int = 5

    # ===================================================================
    # Enhanced Directory Structure with Rotation
    # ===================================================================
    log_dir: str = "logs"
    log_level: str = "Debug"
    log_rotation_lines: int = 2000  # Mandatory 2000-line rotation
    checkpoint_dir: str = "checkpoints"
    model_dir: str = "models"
    tensorboard_dir: str = "logs/tensorboard"

    # InfoBus-specific directories
    info_bus_log_dir: str = "logs/info_bus"
    audit_log_dir: str = "logs/audit"
    operator_log_dir: str = "logs/operator"

    def __post_init__(self) -> None:
        """Post-initialization setup with clamps, dirs, and invariants."""
        object.__setattr__(self, "max_steps_per_episode", int(self.max_steps))

        # ═══════════════════════════════════════════════════════════════
        # AUTO-SET TRADING MODE based on live_mode flag
        # This propagates to all mode-aware subsystems (gates, voting, rewards)
        # ═══════════════════════════════════════════════════════════════
        try:
            from modules.core.trading_mode import TradingModeManager
            TradingModeManager.from_config(self, silent=True)
        except ImportError:
            pass  # Module not available yet during early init

        # Sanitize lists
        self.instruments = [str(x) for x in (self.instruments or []) if str(x).strip()]
        self.timeframes = [str(x) for x in (self.timeframes or []) if str(x).strip()]

        # Clamp ratios/probabilities
        self.max_drawdown = _clamp(float(self.max_drawdown), 0.0, 0.95)
        self.max_total_exposure = _clamp(float(self.max_total_exposure), 0.0, 5.0)
        self.max_position_pct = _clamp(float(self.max_position_pct), 0.0, 1.0)
        self.min_intensity = _clamp(float(self.min_intensity), 0.0, 1.0)
        self.consensus_min = _clamp(float(self.consensus_min), 0.0, 1.0)
        self.consensus_max = _clamp(float(self.consensus_max), 0.0, 1.0)
        if self.consensus_min > self.consensus_max:
            self.consensus_min, self.consensus_max = self.consensus_max, self.consensus_min
        self.emergency_close_threshold = _clamp(float(self.emergency_close_threshold), 0.0, 1.0)
        self.trail_pct = _clamp(float(self.trail_pct), 0.0, 1.0)
        self.confidence_decay = _clamp(float(self.confidence_decay), 0.0, 1.0)

    # Timers/intervals
        self.info_bus_init_timeout = max(0.0, float(self.info_bus_init_timeout))
        self.orchestrator_init_timeout = max(0.0, float(self.orchestrator_init_timeout))
        # Clamp small sync wait (non-blocking feel). Set 0 to fully disable waiting.
        self.orchestrator_sync_wait_ms = max(0.0, float(self.orchestrator_sync_wait_ms))
        # Backpressure / pacing
        try:
            self.orchestrator_max_inflight = max(1, int(self.orchestrator_max_inflight))
        except Exception:
            self.orchestrator_max_inflight = 1
        try:
            self.orchestrator_step_interval = max(1, int(self.orchestrator_step_interval))
        except Exception:
            self.orchestrator_step_interval = 1
        self.risk_check_frequency = max(1, int(self.risk_check_frequency))
        self.risk_alert_cooldown = max(0, int(self.risk_alert_cooldown))
        self.max_concurrent_alerts = max(1, int(self.max_concurrent_alerts))
        self.max_steps = max(1, int(self.max_steps))
        self.max_steps_per_episode = self.max_steps  # keep alias in sync
        self.environment_observation_size = max(1, int(self.environment_observation_size))

        # Ensure directories exist
        all_dirs = [
            self.log_dir, self.model_dir,
            self.tensorboard_dir, self.data_dir, self.info_bus_log_dir,
            self.audit_log_dir, self.operator_log_dir
        ]
        for directory in all_dirs:
            _ensure_dir(directory)

        # Create module-specific log directories
        module_log_dirs = [
            "logs/trading", "logs/risk", "logs/strategy", "logs/memory",
            "logs/voting", "logs/position", "logs/features", "logs/meta"
        ]
        for directory in module_log_dirs:
            _ensure_dir(directory)

    # ------------------------------------------------------------------
    # Structured views
    # ------------------------------------------------------------------
    def get_model_config(self) -> Dict[str, Any]:
        """Legacy PPO model defaults (agents should own live HPs)."""
        return {
            "learning_rate": float(self.learning_rate),
            "n_steps": int(self.n_steps),
            "batch_size": int(self.batch_size),
            "n_epochs": int(self.n_epochs),
            "gamma": float(self.gamma),
            "gae_lambda": float(self.gae_lambda),
            "clip_range": float(self.clip_range),
            "clip_range_vf": (None if self.clip_range_vf is None else float(self.clip_range_vf)),
            "ent_coef": float(self.ent_coef),
            "vf_coef": float(self.vf_coef),
            "max_grad_norm": float(self.max_grad_norm),
            "target_kl": (None if self.target_kl is None else float(self.target_kl)),
            "policy_hidden_size": int(self.policy_hidden_size),
            "value_hidden_size": int(self.value_hidden_size),
        }

    def get_info_bus_config(self) -> Dict[str, Any]:
        """InfoBus configuration dictionary."""
        return {
            "enabled": bool(self.info_bus_enabled),
            "audit_level": str(self.info_bus_audit_level),
            "validation": bool(self.info_bus_validation),
            "log_dir": str(self.info_bus_log_dir),
            "rotation_lines": int(self.log_rotation_lines),
        }

    def get_bus_policy(self) -> Dict[str, Any]:
        """Bus-first policy and alias map for env/trainer wiring."""
        return {
            "bus_first": bool(self.bus_first),
            "prefer_bus_data": bool(self.prefer_bus_data),
            "prefer_bus_features": bool(self.prefer_bus_features),
            "prefer_bus_rewards": bool(self.prefer_bus_rewards),
            "prefer_bus_metrics": bool(self.prefer_bus_metrics),
            "prefer_bus_limits": bool(self.prefer_bus_limits),
            "allow_module_overrides": bool(self.allow_module_overrides),
            "halt_on_emergency": bool(self.halt_on_emergency),
            "aliases": {k: list(v) for k, v in (self.bus_aliases or {}).items()},
        }

    def get_module_config(self) -> Dict[str, Any]:
        """Module configuration defaults."""
        return {
            "debug": bool(self.debug),
            "max_history": 100,
            "audit_enabled": True,
            "log_rotation_lines": int(self.log_rotation_lines),
            "health_check_interval": 100,
            "info_bus_enabled": bool(self.info_bus_enabled),
        }

    # ------------------------------------------------------------------
    # (De)serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict of this config (converts complex types to builtins)."""
        return _to_builtin(asdict(self))

    def save_config(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> "TradingConfig":
        """Load configuration from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        clean = {k: v for k, v in config_dict.items() if k in allowed}
        return cls(**clean)

    # ------------------------------------------------------------------
    # Ergonomics
    # ------------------------------------------------------------------
    def clone_with(self, **overrides: Any) -> "TradingConfig":
        """Return a copy with selected fields overridden."""
        known = {k: v for k, v in overrides.items() if hasattr(self, k)}
        return replace(self, **known)

    def apply_overrides(self, **overrides: Any) -> None:
        """In-place override for known fields (keeps compatibility)."""
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        self.__post_init__()  # re-run clamps/dirs when critical fields changed

    def __str__(self) -> str:
        """String representation for logging."""
        return (
            f"TradingConfig(\n"
            f"  Mode: {'LIVE' if self.live_mode else ('TEST' if self.test_mode else 'BACKTEST')}\n"
            f"  InfoBus: {'ENABLED' if self.info_bus_enabled else 'DISABLED'}  (bus-first: {self.bus_first})\n"
            f"  Balance: ${self.initial_balance:,.2f}\n"
            f"  Max Steps: {self.max_steps}\n"
            f"  Instruments: {self.instruments}\n"
            f"  Obs Size: {self.environment_observation_size}\n"
            f"  Training Steps: {self.final_training_steps:,}\n"
            f"  Risk Limits (fallback): DD={self.max_drawdown:.1%}, Exposure={self.max_total_exposure:.1%}, Pos={self.max_position_pct:.1%}\n"
            f"  Log Rotation: {self.log_rotation_lines} lines\n"
            f")"
        )


# ─────────────────────────────────────────────────────────
# Runtime market/episode state (unchanged)
# ─────────────────────────────────────────────────────────
@dataclass
class MarketState:
    """Enhanced market state with InfoBus integration"""
    balance: float
    peak_balance: float
    current_step: int
    current_drawdown: float
    last_trade_step: Dict[str, int] = field(default_factory=dict)

    # Enhanced state tracking
    session_start_balance: float = field(init=False)
    session_trades: int = 0
    session_pnl: float = 0.0
    last_info_bus_update: int = 0

    def __post_init__(self):
        object.__setattr__(self, "session_start_balance", float(self.balance))


@dataclass
class EpisodeMetrics:
    """Enhanced episode metrics with InfoBus tracking"""
    pnls: List[float] = field(default_factory=list)
    durations: List[int] = field(default_factory=list)
    drawdowns: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    votes_log: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_trace: List[str] = field(default_factory=list)

    # InfoBus-specific metrics
    info_bus_events: List[Dict[str, Any]] = field(default_factory=list)
    module_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    consensus_history: List[float] = field(default_factory=list)
    risk_alerts: List[Dict[str, Any]] = field(default_factory=list)


# ─────────────────────────────────────────────────────────
# Presets (set bus-first flags to True by default)
# ─────────────────────────────────────────────────────────
class ConfigPresets:
    """Enhanced preset configurations for InfoBus-integrated environment"""

    @staticmethod
    def conservative_live() -> TradingConfig:
        """Ultra-conservative configuration for LIVE trading with real money."""
        return TradingConfig(
            # ═══════════════════════════════════════════════════════════════
            # ULTRA-CONSERVATIVE RISK SETTINGS FOR LIVE TRADING
            # ═══════════════════════════════════════════════════════════════
            initial_balance=1000.0,
            max_position_pct=0.03,         # Max 3% per position (was 5%)
            max_total_exposure=0.10,       # Max 10% total exposure (was 15%)
            max_drawdown=0.08,             # Max 8% drawdown before halt (was 10%)
            max_correlation=0.6,           # Lower correlation tolerance
            
            # Strong consensus requirements
            consensus_min=0.65,            # Need 65% agreement (was 50%)
            min_confidence=0.50,           # Need 50% confidence
            min_intensity=0.40,            # Need strong signal
            
            # Position management - very conservative
            max_consecutive_losses=3,      # Only 3 losses before reducing
            loss_reduction=0.5,            # Reduce by 50% after consecutive losses
            emergency_drawdown_trigger=0.06,  # Emergency at 6% DD
            emergency_close_threshold=0.80,   # Close at 80% risk threshold

            # Live trading settings
            live_mode=True,
            debug=False,
            enable_shadow_sim=False,

            # Bus-first policy (let modules control)
            bus_first=True,
            prefer_bus_data=True,
            prefer_bus_features=True,
            prefer_bus_rewards=True,
            prefer_bus_metrics=True,
            prefer_bus_limits=True,
            allow_module_overrides=True,
            halt_on_emergency=True,

            # InfoBus settings
            info_bus_enabled=True,
            info_bus_audit_level="WARNING",
            info_bus_validation=True,

            # Frequent monitoring
            risk_check_frequency=1,        # Check every step
            risk_alert_cooldown=2,         # Quick alerts
            max_concurrent_alerts=3,       # Fewer alerts before action

            # Conservative learning (shouldn't update live, but safety)
            learning_rate=1e-5,            # Very slow learning
            ent_coef=0.001,
            n_steps=512,

            # Short episodes for quick recovery
            max_steps=50,
            final_training_steps=10000,

            # Logging
            log_interval=1,                # Log every step
            checkpoint_freq=1000,
            eval_freq=500,

            # Single instrument to start
            instruments=["EUR_USD"],
            timeframes=["H1", "H4", "D1"],
        )

    @staticmethod
    def research_mode() -> TradingConfig:
        return TradingConfig(
            initial_balance=5000.0,
            max_position_pct=0.10,
            max_total_exposure=0.30,
            max_drawdown=0.25,

            live_mode=False,
            test_mode=True,
            debug=True,
            enable_shadow_sim=True,
            enable_news_sentiment=True,

            # Bus-first policy
            bus_first=True,
            prefer_bus_data=True,
            prefer_bus_features=True,
            prefer_bus_rewards=True,
            prefer_bus_metrics=True,
            prefer_bus_limits=True,
            allow_module_overrides=True,
            halt_on_emergency=True,

            info_bus_enabled=True,
            info_bus_audit_level="DEBUG",
            info_bus_validation=True,

            risk_check_frequency=1,
            risk_alert_cooldown=1,
            max_concurrent_alerts=20,

            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,

            max_steps=100,
            final_training_steps=25000,

            log_interval=1,
            checkpoint_freq=5000,
            eval_freq=2500,

            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["H1", "H4", "D1"],
        )

    @staticmethod
    def production_backtest() -> TradingConfig:
        return TradingConfig(
            initial_balance=10000.0,
            max_position_pct=0.15,
            max_total_exposure=0.40,
            max_drawdown=0.25,
            consensus_min=0.30,

            live_mode=False,
            test_mode=False,
            debug=False,
            enable_shadow_sim=True,

            # Bus-first policy
            bus_first=True,
            prefer_bus_data=True,
            prefer_bus_features=True,
            prefer_bus_rewards=True,
            prefer_bus_metrics=True,
            prefer_bus_limits=True,
            allow_module_overrides=True,
            halt_on_emergency=True,

            info_bus_enabled=True,
            info_bus_audit_level="DEBUG",
            info_bus_validation=True,

            risk_check_frequency=1,
            risk_alert_cooldown=5,
            max_concurrent_alerts=10,

            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,

            max_steps=200,
            final_training_steps=100000,

            log_interval=10,
            checkpoint_freq=10000,
            eval_freq=5000,

            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["H1", "H4", "D1"],
        )


# ─────────────────────────────────────────────────────────
# Factory & Validation
# ─────────────────────────────────────────────────────────
class ConfigFactory:
    """Factory for creating InfoBus-compatible configurations (bus-first)."""

    @staticmethod
    def create_config(
        mode: str = "backtest",
        risk_level: str = "moderate",
        info_bus_level: str = "auto",
        **overrides: Any
    ) -> TradingConfig:
        """
        Create a TradingConfig with appropriate mode settings.
        
        Args:
            mode: "live", "research", "production", or "backtest"
            risk_level: "conservative", "moderate", or "aggressive"
            info_bus_level: "auto", "DEBUG", "INFO", "WARNING", "ERROR"
            **overrides: Additional config overrides
        
        Note: Setting mode="live" automatically enables LIVE trading mode
              across all subsystems (gates, voting, rewards).
        """
        if mode == "live":
            config = ConfigPresets.conservative_live()
        elif mode == "research":
            config = ConfigPresets.research_mode()
        elif mode == "production":
            config = ConfigPresets.production_backtest()
        else:
            config = TradingConfig()

        # Adjust InfoBus level
        if info_bus_level == "auto":
            if config.debug:
                config.info_bus_audit_level = "DEBUG"
            elif config.live_mode:
                config.info_bus_audit_level = "WARNING"
            else:
                config.info_bus_audit_level = "INFO"
        else:
            config.info_bus_audit_level = str(info_bus_level).upper()

        # Risk level (affects only fallback guard-rails)
        if risk_level == "conservative":
            config.max_position_pct *= 0.5
            config.max_total_exposure *= 0.7
            config.max_drawdown *= 0.8
            config.risk_check_frequency = 1
            config.risk_alert_cooldown = 3
        elif risk_level == "aggressive":
            config.max_position_pct *= 1.5
            config.max_total_exposure *= 1.3
            config.max_drawdown *= 1.2
            config.risk_check_frequency = 2
            config.risk_alert_cooldown = 10

        # Apply overrides (known fields only)
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}'")

        config.__post_init__()
        
        # Explicitly set trading mode after config is fully built
        try:
            from modules.core.trading_mode import TradingModeManager
            TradingModeManager.from_config(config)
        except ImportError:
            pass
        
        return config


def validate_config(config: TradingConfig) -> List[str]:
    """Validation with InfoBus compatibility checks (config = fallback only)."""
    warnings: List[str] = []

    # If bus-first but bus disabled → warn
    if config.bus_first and not config.info_bus_enabled:
        warnings.append("⚠️ bus_first is True but InfoBus is disabled; falling back to static config.")

    # Standard guard-rail validation (only meaningful when bus data absent)
    if config.max_total_exposure > 1.0:
        warnings.append("⚠️ Fallback total exposure > 100% is very risky")
    if config.max_drawdown > 0.5:
        warnings.append("⚠️ Fallback max drawdown > 50% is extremely risky")
    if config.max_position_pct > 0.3:
        warnings.append("⚠️ Fallback position size > 30% per trade is very risky")

    # InfoBus validation
    if config.info_bus_enabled:
        if config.info_bus_audit_level not in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            warnings.append("⚠️ Invalid InfoBus audit level")
        if config.risk_check_frequency < 1:
            warnings.append("⚠️ Risk check frequency too low for InfoBus")
        if config.log_rotation_lines > 5000:
            warnings.append("⚠️ Log rotation lines > 5000 may impact performance")

    # Live mode hints
    if config.live_mode and not config.info_bus_enabled:
        warnings.append("⚠️ InfoBus strongly recommended for live trading")

    return warnings


# ===================================================================
# Example Usage (manual test)
# ===================================================================
if __name__ == "__main__":
    print("🔧 Enhanced BUS-FIRST Trading Configuration System")
    print("=" * 72)

    configs = {
        "Conservative Live": ConfigPresets.conservative_live(),
        "Research Mode": ConfigPresets.research_mode(),
        "Production Backtest": ConfigPresets.production_backtest(),
    }

    for name, config in configs.items():
        print(f"\n📋 {name}:")
        print(f"  InfoBus: {config.info_bus_enabled} ({config.info_bus_audit_level}) | bus_first={config.bus_first}")
        print(f"  Risk Fallbacks: DD={config.max_drawdown:.1%}, Exposure={config.max_total_exposure:.1%}, Pos={config.max_position_pct:.1%}")
        print(f"  Obs Size: {config.environment_observation_size}")
        warns = validate_config(config)
        if warns:
            print(f"  ⚠️ Warnings: {len(warns)}")
            for warning in warns[:3]:
                print(f"    - {warning}")

    print("\n✅ BUS-FIRST configuration system test completed!")
