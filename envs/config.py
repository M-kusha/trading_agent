# envs/config.py
"""
Enhanced Configuration System for InfoBus-Integrated Trading Environment
BUS-FIRST edition: modules are source-of-truth; config is fallback + guard-rails.

Key ideas:
- InfoBus + modules are the canonical owners of risk, limits, rewards, features.
- TradingConfig provides sane defaults and guard-rails if the bus has no data.
- ModernTradingEnv is bus-first and uses this config purely as a fallback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional
from pathlib import Path

# Import canonical timeframe constants (single source of truth)
try:
    from modules.voting.core.constants import PRIMARY_TIMEFRAME
except ImportError:
    PRIMARY_TIMEFRAME = "M15"  # Fallback if voting module not available


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
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
    initial_balance: float = 100_000.0  # 100k for prop firm simulation
    max_steps: int = 100_000           # increase default episode length to reduce frequent resets
    debug: bool = True
    init_seed: int = 42
    max_steps_per_episode: int = field(init=False)

    # Observation sizing:
    # - ModernTradingEnv uses PPO_OBS_SIZE (48) from ppo_observation_builder as the
    #   canonical size.
    # - This field is kept for external tooling / sanity checks and is clamped in __post_init__.
    environment_observation_size: int = 48  # Must match PPO_OBS_SIZE from ppo_observation_builder

    # Primary timeframe: M15 is the single decision/execution timeframe.
    # H1/H4/D1 are context timeframes only. Uses canonical constant from voting module.
    primary_timeframe: str = PRIMARY_TIMEFRAME

    # Minimum data bars required (prevents tiny datasets producing trivial episodes)
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
    # so decision modules can enqueue orders before the env collects intents.
    orchestrator_sync_wait_ms: float = 25.0

    # Limit the number of concurrent orchestrator executions scheduled by the env
    orchestrator_max_inflight: int = 1

    # Only schedule orchestrator once every N env steps (1 = every step)
    orchestrator_step_interval: int = 1

    # Optional general step throttle (ms) applied even when orchestrator is disabled
    # or when no orchestrator scheduling occurs on a given step. Default 0 (no delay).
    step_sleep_ms: float = 0.0

    # Bus-first policy toggles (modules via SmartInfoBus are single source of truth)
    bus_first: bool = True
    prefer_bus_data: bool = True          # prefer MarketDataProvider over local data
    prefer_bus_features: bool = True      # prefer AdvancedFeatureEngine/MultiScaleFeatureEngine
    prefer_bus_rewards: bool = True       # prefer RiskAdjustedReward.shaped_reward
    prefer_bus_metrics: bool = True       # prefer PortfolioRiskSystem/DrawdownRescue/etc
    prefer_bus_limits: bool = True        # prefer Compliance/PortfolioRiskSystem/etc
    allow_module_overrides: bool = True   # let modules override defaults at runtime
    halt_on_emergency: bool = True        # stop on emergency/risk kill switch

    # Canonical alias map for SmartInfoBus wiring (modules declare owners)
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
        "account_state": ["Executor"],

        # Features / observations
        "advanced_features": ["AdvancedFeatureEngine", "MultiScaleFeatureEngine"],
        "environment_observation": ["Environment"],

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
        "volatility_data": ["MarketDataProvider"],
        "trading_session": ["MarketDataProvider"],

        # Prop firm status & memory (context used by env obs builder)
        "prop_firm_status": ["PropFirmGuard"],
        "memory_gate": ["UnifiedMemory"],
        "memory_vote": ["UnifiedMemory"],
        "neural_risk_hint": ["UnifiedMemory"],
    })

    # ===================================================================
    # Data and Instruments (used as defaults/fallbacks)
    # ===================================================================
    data_dir: str = "data/processed"
    instruments: List[str] = field(default_factory=lambda: ["EUR_USD", "XAU_USD"])
    timeframes: List[str] = field(default_factory=lambda: ["M15", "H1", "H4", "D1"])

    # ===================================================================
    # Trading Parameters (fallback only; modules own live values)
    # ===================================================================
    no_trade_penalty: float = 0.2          # Reduced - don't force trading
    consensus_min: float = 0.50            # Raised from 0.30 - need more agreement
    consensus_max: float = 0.85            # Raised from 0.70 - higher ceiling
    max_episodes: int = 10_000

    # Execution economics (fallbacks; PositionManager/EQM should override via bus)
    default_spread: float = 0.0
    slippage_pts: float = 0.0
    commission_per_million: float = 0.0

    # Soft gating for env-embedded logic (fallback only; modules should own these live)
    min_confidence: float = 0.35           # require confidence
    min_intensity: float = 0.35            # filter weak signals
    ignore_hold: bool = True

    # ===================================================================
    # Risk Management (fallback guard-rails; Compliance/PortfolioRiskSystem are canonical)
    # ===================================================================
    rotation_gap: int = 5
    max_position_pct: float = 0.15       # smaller positions = less risk
    max_total_exposure: float = 0.35     # less total exposure
    max_drawdown: float = 0.15           # tighter DD limit
    max_correlation: float = 0.7

    profit_target: float = 0.10          # From prop_firm.profit_target (10% default)

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
    emergency_drawdown_trigger: float = 0.15    # trigger if current drawdown > 15%
    emergency_exposure_trigger: float = 0.40    # trigger if exposure > 40% (prop firm friendly)
    emergency_risk_score_threshold: float = 0.7 # require risk_score >= 0.7 to escalate
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
    num_envs: int = 16
    test_mode: bool = False
    live_mode: bool = False
    training_mode: bool = False
    enable_shadow_sim: bool = True
    enable_news_sentiment: bool = False

    # Module Enablement Flags (coarse – mostly for orchestrator wiring)
    enable_meta_rl: bool = True
    enable_memory_systems: bool = True
    enable_strategy_evolution: bool = True
    enable_risk_monitoring: bool = True
    enable_visualization: bool = True

    # ===================================================================
    # PPO Hyperparameters (tuned for trading - v4.1 Autonomous PPO)
    # ===================================================================
    # NOTE: Fine-tuned for M15-PRIMARY short-term trading:
    # - M15 = primary signal timeframe (15-min bars)
    # - ExitEngine closes positions after ~1.5-4 hours (time_decay_hours=1.5)
    # - Tight trailing TP means H1/H4/D1 trends rarely play out
    # - gamma=0.95 → horizon ~20 steps = ~5 hours = matches exit horizon
    learning_rate: float = 1e-4          # Reduced from 3e-4 for stability
    n_steps: int = 2048                  # Keep - good for trading episodes
    batch_size: int = 128                # Increased from 64 for better gradients
    n_epochs: int = 10                   # Keep - good balance
    gamma: float = 0.95                  # SHORT-TERM: ~5h horizon matches ExitEngine timeouts
    gae_lambda: float = 0.95             # Keep - standard
    clip_range: float = 0.15             # Reduced from 0.2 - more conservative updates
    clip_range_vf: Optional[float] = None
    ent_coef: float = 0.02               # Increased from 0.01 - encourage exploration
    vf_coef: float = 0.5                 # Keep - standard
    max_grad_norm: float = 0.5           # Keep - standard
    target_kl: Optional[float] = 0.015   # Slightly higher for more exploration
    
    # Direction thresholds for interpreting PPO action[0] (v4.1)
    direction_long_threshold: float = 0.3   # action[0] > this = LONG
    direction_short_threshold: float = -0.3  # action[0] < this = SHORT
    
    # Autonomous PPO training mode (v4.1)
    # When True, PPO direction is used directly without expert blending
    ppo_autonomous_training: bool = True

    # ===================================================================
    # Network Architecture (legacy defaults; agent modules own live nets)
    # ===================================================================
    policy_hidden_size: int = 256
    value_hidden_size: int = 256

    # ===================================================================
    # Training Schedule (env/trainer level schedules)
    # ===================================================================
    final_training_steps: int = 100_000
    log_interval: int = 10
    checkpoint_freq: int = 10_000
    eval_freq: int = 20_000
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

    audit_log_dir: str = "logs/audit"
    operator_log_dir: str = "logs/operator"

    # External risk configuration payloads (injected by central config layer)
    risk_policy_payload: Optional[Dict[str, Any]] = None
    risk_overrides_payload: Optional[Dict[str, Any]] = None

    # Internal flag: ensure risk_policy.yaml is only loaded once per instance
    _risk_policy_loaded: bool = field(init=False, default=False, repr=False)

    # ─────────────────────────────────────────────────────────
    # Post-init
    # ─────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        """Post-initialization setup with clamps, dirs, mode, and invariants."""
        object.__setattr__(self, "max_steps_per_episode", int(self.max_steps))

        # Load risk policy from injected payload or fallback file
        self._load_from_risk_policy(self.risk_policy_payload)

        # AUTO-SET TRADING MODE based on live_mode flag
        # Only upgrade to LIVE; never downgrade from LIVE to TRAINING.
        try:
            from modules.core.trading_mode import TradingModeManager

            if self.live_mode:
                TradingModeManager.set_mode("LIVE", silent=True)
            elif not TradingModeManager.is_live():
                TradingModeManager.set_mode("TRAINING", silent=True)
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
        self.orchestrator_sync_wait_ms = max(0.0, float(self.orchestrator_sync_wait_ms))

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
        self.max_steps_per_episode = self.max_steps
        self.environment_observation_size = max(1, int(self.environment_observation_size))

        # Ensure directories exist
        all_dirs = [
            self.log_dir,
            self.model_dir,
            self.tensorboard_dir,
            self.data_dir,
            self.audit_log_dir,
            self.operator_log_dir,
        ]
        for directory in all_dirs:
            _ensure_dir(directory)

        # Create module-specific log directories
        module_log_dirs = [
            "logs/trading",
            "logs/risk",
            "logs/strategy",
            "logs/memory",
            "logs/voting",
            "logs/position",
            "logs/meta",
        ]
        for directory in module_log_dirs:
            _ensure_dir(directory)

    # ─────────────────────────────────────────────────────────
    # risk_policy.yaml integration
    # ─────────────────────────────────────────────────────────
    def _load_from_risk_policy(self, payload: Optional[Dict[str, Any]] = None) -> None:
        """
        Load risk parameters from config/risk_policy.yaml.

        Important:
        - This is only applied ONCE per TradingConfig instance.
        - Subsequent __post_init__ calls (e.g. via apply_overrides / factory)
          will not re-load or overwrite with risk_policy values.
        """
        if getattr(self, "_risk_policy_loaded", False):
            return

        try:
            cfg: Dict[str, Any] = {}
            if payload is not None:
                cfg = dict(payload)
            else:
                import yaml

                risk_policy_path = Path("config/risk_policy.yaml")
                if not risk_policy_path.exists():
                    object.__setattr__(self, "_risk_policy_loaded", True)
                    return

                with open(risk_policy_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}

            prop_firm = cfg.get("prop_firm", {})
            lot_sizing = cfg.get("lot_sizing", {})
            limits = cfg.get("limits", {})
            position_manager = cfg.get("position_manager", {})
            smart_position = cfg.get("smart_position", {})
            escalation = cfg.get("escalation", {})

            # Account balance (prop_firm.account_size or lot_sizing.account_balance)
            balance = prop_firm.get("account_size") or lot_sizing.get("account_balance")
            if balance and float(balance) > 0:
                self.initial_balance = float(balance)

            # Risk limits
            if "max_daily_loss" in limits:
                # Use daily loss as max_drawdown (more conservative)
                self.max_drawdown = float(limits["max_daily_loss"])
            elif "max_drawdown" in limits:
                self.max_drawdown = float(limits["max_drawdown"])

            if "max_position_size" in limits:
                self.max_position_pct = float(limits["max_position_size"])

            if "max_exposure_pct" in limits:
                self.max_total_exposure = float(limits["max_exposure_pct"])

            if "max_correlation" in limits:
                self.max_correlation = float(limits["max_correlation"])

            # Position manager settings
            if "max_consecutive_losses" in position_manager:
                self.max_consecutive_losses = int(position_manager["max_consecutive_losses"])

            if "emergency_drawdown_trigger" in position_manager:
                self.emergency_drawdown_trigger = float(position_manager["emergency_drawdown_trigger"])

            if "emergency_exposure_trigger" in position_manager:
                self.emergency_exposure_trigger = float(position_manager["emergency_exposure_trigger"])

            # Smart position settings
            if "hard_stop_loss_eur" in smart_position:
                self.hard_loss_eur = float(smart_position["hard_stop_loss_eur"])

            if "min_signal_strength" in smart_position:
                self.min_signal_threshold = float(smart_position["min_signal_strength"])

            if "profit_take_trail_pct" in smart_position:
                self.trail_pct = float(smart_position["profit_take_trail_pct"])

            # Escalation thresholds
            if "shutdown_threshold" in escalation:
                self.emergency_close_threshold = float(escalation["shutdown_threshold"])

            # Profit target from prop firm
            if "profit_target" in prop_firm:
                self.profit_target = float(prop_firm["profit_target"])

            # Explicit overrides from central config (last-write-wins)
            if self.risk_overrides_payload:
                ro = self.risk_overrides_payload
                if "max_total_exposure" in ro and ro["max_total_exposure"] is not None:
                    self.max_total_exposure = float(ro["max_total_exposure"])
                if "max_position_pct" in ro and ro["max_position_pct"] is not None:
                    self.max_position_pct = float(ro["max_position_pct"])
                if "max_drawdown" in ro and ro["max_drawdown"] is not None:
                    self.max_drawdown = float(ro["max_drawdown"])
                if "emergency_drawdown_trigger" in ro and ro["emergency_drawdown_trigger"] is not None:
                    self.emergency_drawdown_trigger = float(ro["emergency_drawdown_trigger"])

        except Exception:
            # Keep defaults if config load fails
            pass
        finally:
            object.__setattr__(self, "_risk_policy_loaded", True)

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
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> "TradingConfig":
        """
        Load configuration from JSON file.

        Notes:
            - Only fields with init=True are passed to the constructor.
            - Internal fields like `_risk_policy_loaded` and
              `max_steps_per_episode` are reconstructed in __post_init__.
        """
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        init_fields = {
            name
            for name, f in cls.__dataclass_fields__.items()  # type: ignore[attr-defined]
            if getattr(f, "init", True)
        }
        clean = {k: v for k, v in config_dict.items() if k in init_fields}
        return cls(**clean)

    # ------------------------------------------------------------------
    # Ergonomics
    # ------------------------------------------------------------------
    def clone_with(self, **overrides: Any) -> "TradingConfig":
        """Return a copy with selected fields overridden."""
        known = {k: v for k, v in overrides.items() if hasattr(self, k)}
        return replace(self, **known)

    def apply_overrides(self, **overrides: Any) -> None:
        """
        In-place override for known fields (keeps compatibility).

        Note: risk_policy.yaml will NOT be re-applied (it is loaded once).
        """
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
        # Re-run clamps/dirs/mode; _load_from_risk_policy() will no-op if already loaded
        self.__post_init__()

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
            f"  Risk Limits (fallback): DD={self.max_drawdown:.1%}, "
            f"Exposure={self.max_total_exposure:.1%}, Pos={self.max_position_pct:.1%}\n"
            f"  Log Rotation: {self.log_rotation_lines} lines\n"
            f")"
        )


# ─────────────────────────────────────────────────────────
# Runtime market/episode state (unchanged, used by env)
# ─────────────────────────────────────────────────────────
@dataclass
class MarketState:
    """Enhanced market state with InfoBus integration."""
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_start_balance", float(self.balance))


@dataclass
class EpisodeMetrics:
    """Enhanced episode metrics with InfoBus tracking."""
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
# Presets (bus-first by default except exploration_mode)
# ─────────────────────────────────────────────────────────
class ConfigPresets:
    """Preset configurations for InfoBus-integrated environment."""

    @staticmethod
    def exploration_mode() -> TradingConfig:
        """
        NO MODULES configuration for initial exploration/pretraining.

        Use this to train the agent on raw price data BEFORE adding modules.
        The env runs without InfoBus / modules; all logic is local.
        """
        return TradingConfig(
            # Disable all module/bus features – pure exploration
            bus_first=False,
            prefer_bus_data=False,
            prefer_bus_features=False,
            prefer_bus_rewards=False,
            prefer_bus_metrics=False,
            prefer_bus_limits=False,
            allow_module_overrides=False,
            halt_on_emergency=False,

            # Disable InfoBus & Orchestrator
            info_bus_enabled=False,
            info_bus_validation=False,
            orchestrator_init_timeout=0.0,
            orchestrator_sync_wait_ms=0.0,

            # Permissive risk settings for exploration
            # (initial_balance will still be loaded from risk_policy.yaml if present)
            initial_balance=100_000.0,
            max_position_pct=0.20,
            max_total_exposure=0.50,
            max_drawdown=0.30,

            # No consensus requirements (agent acts alone)
            consensus_min=0.0,
            consensus_max=1.0,
            min_confidence=0.0,
            min_intensity=0.0,

            # PPO training settings tuned for exploration
            # NOTE: gamma=0.95 for M15 short-term trading (5h horizon matches ExitEngine)
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.95,             # SHORT-TERM: ~5h horizon for M15 trading
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,

            # Network (smaller for faster exploration)
            policy_hidden_size=128,
            value_hidden_size=128,

            # Environment
            max_steps=10_000,
            environment_observation_size=48,  # keep aligned with PPO_OBS_SIZE
            min_required_data_bars=50,

            # Training schedule
            final_training_steps=100_000,
            checkpoint_freq=10_000,
            eval_freq=5_000,
            n_eval_episodes=5,

            # Mode flags
            live_mode=False,
            test_mode=False,
            debug=True,

            # Data
            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["M15", "H1", "H4", "D1"],

            # No execution costs for clean exploration
            default_spread=0.0,
            slippage_pts=0.0,
            commission_per_million=0.0,
        )

    @staticmethod
    def conservative_live() -> TradingConfig:
        """Ultra-conservative configuration for LIVE trading with real money."""
        return TradingConfig(
            # Ultra-conservative risk settings
            initial_balance=1_000.0,
            max_position_pct=0.03,
            max_total_exposure=0.10,
            max_drawdown=0.08,
            max_correlation=0.6,

            # Strong consensus requirements
            consensus_min=0.65,
            min_confidence=0.50,
            min_intensity=0.40,

            # Position management - very conservative
            max_consecutive_losses=3,
            loss_reduction=0.5,
            emergency_drawdown_trigger=0.06,
            emergency_close_threshold=0.80,

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
            risk_check_frequency=1,
            risk_alert_cooldown=2,
            max_concurrent_alerts=3,

            # Conservative learning (safety; real live trading should be frozen anyway)
            learning_rate=1e-5,
            ent_coef=0.001,
            n_steps=512,

            # Short episodes for quick recovery
            max_steps=50,
            final_training_steps=10_000,

            # Logging
            log_interval=1,
            checkpoint_freq=1_000,
            eval_freq=500,

            # Start live with a single instrument
            instruments=["EUR_USD"],
            timeframes=["M15", "H1", "H4", "D1"],
        )

    @staticmethod
    def research_mode() -> TradingConfig:
        """Bus-first research configuration for heavy diagnostics and module work."""
        return TradingConfig(
            initial_balance=5_000.0,
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
            final_training_steps=25_000,

            log_interval=1,
            checkpoint_freq=5_000,
            eval_freq=2_500,

            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["M15", "H1", "H4", "D1"],
        )

    @staticmethod
    def production_backtest() -> TradingConfig:
        """Bus-first production-style backtest for full committee + RL."""
        return TradingConfig(
            initial_balance=10_000.0,
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
            final_training_steps=100_000,

            log_interval=10,
            checkpoint_freq=10_000,
            eval_freq=5_000,

            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["M15", "H1", "H4", "D1"],
        )

    @staticmethod
    def training_fast() -> TradingConfig:
        """
        High-speed PPO training preset - optimized for maximum steps/second.
        
        Key optimizations:
        - training_mode=True: Modules know we're in RL learning mode
        - Minimal logging (ERROR level only)
        - No validation overhead on InfoBus
        - No sync waits between steps
        - Shadow sim and visualization disabled
        
        Expected: 40-80 steps/sec (vs ~13 with production preset)
        """
        return TradingConfig(
            # ═══════════════════════════════════════════════════════════
            # CRITICAL: Runtime mode flags
            # ═══════════════════════════════════════════════════════════
            training_mode=True,       # <- CRITICAL: modules know we're in RL mode
            live_mode=False,
            test_mode=False,
            debug=False,

            # ═══════════════════════════════════════════════════════════
            # Trading parameters
            # ═══════════════════════════════════════════════════════════
            initial_balance=100_000.0,
            max_position_pct=0.15,
            max_total_exposure=0.40,
            max_drawdown=0.25,
            consensus_min=0.30,

            # ═══════════════════════════════════════════════════════════
            # InfoBus: enabled but CHEAP
            # ═══════════════════════════════════════════════════════════
            info_bus_enabled=True,
            info_bus_audit_level="ERROR",    # Only errors, no DEBUG spam
            info_bus_validation=False,       # Skip schema checks every step
            info_bus_init_timeout=2.0,

            # ═══════════════════════════════════════════════════════════
            # Orchestrator: NO WAITING
            # ═══════════════════════════════════════════════════════════
            orchestrator_async_init=False,   # Simple deterministic init
            orchestrator_sync_wait_ms=0.0,   # NO waiting between steps
            orchestrator_step_interval=1,
            step_sleep_ms=0.0,               # No artificial delays

            # ═══════════════════════════════════════════════════════════
            # Bus-first policy (keep logic, minimize overhead)
            # ═══════════════════════════════════════════════════════════
            bus_first=True,
            prefer_bus_data=True,
            prefer_bus_features=True,
            prefer_bus_rewards=True,
            prefer_bus_metrics=True,
            prefer_bus_limits=True,
            allow_module_overrides=True,
            halt_on_emergency=True,

            # ═══════════════════════════════════════════════════════════
            # Features: keep logic, disable eye-candy
            # ═══════════════════════════════════════════════════════════
            enable_shadow_sim=False,         # Use separate script for shadow testing
            enable_visualization=False,      # No plots/HTML during PPO
            enable_risk_monitoring=True,     # Keep risk logic (it's light)
            enable_news_sentiment=False,
            enable_meta_rl=True,             # In-memory logic, fine
            enable_memory_systems=True,
            enable_strategy_evolution=True,

            # ═══════════════════════════════════════════════════════════
            # Risk settings
            # ═══════════════════════════════════════════════════════════
            risk_check_frequency=1,
            risk_alert_cooldown=5,
            max_concurrent_alerts=10,

            # ═══════════════════════════════════════════════════════════
            # PPO hyperparameters
            # ═══════════════════════════════════════════════════════════
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.95,
            gae_lambda=0.95,
            clip_range=0.12,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.015,

            # ═══════════════════════════════════════════════════════════
            # Episode & training limits
            # ═══════════════════════════════════════════════════════════
            max_steps=200,
            final_training_steps=1_000_000,

            # ═══════════════════════════════════════════════════════════
            # Logging: MINIMAL for speed
            # ═══════════════════════════════════════════════════════════
            log_level="Error",               # Only errors
            log_rotation_lines=20000,        # Rotate less often
            log_interval=10,
            checkpoint_freq=10_000,
            eval_freq=5_000,

            # ═══════════════════════════════════════════════════════════
            # Instruments & timeframes
            # ═══════════════════════════════════════════════════════════
            instruments=["EUR_USD", "XAU_USD"],
            timeframes=["M15", "H1", "H4", "D1"],
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
        **overrides: Any,
    ) -> TradingConfig:
        """
        Create a TradingConfig with appropriate mode settings.

        Args:
            mode: "live", "research", "production", or "backtest"
            risk_level: "conservative", "moderate", or "aggressive"
            info_bus_level: "auto", "DEBUG", "INFO", "WARNING", "ERROR"
            **overrides: Additional config overrides

        Note:
            Setting mode="live" automatically enables LIVE trading mode
            across all subsystems (gates, voting, rewards).
        """
        # 1) Base preset
        if mode == "live":
            config = ConfigPresets.conservative_live()
        elif mode == "research":
            config = ConfigPresets.research_mode()
        elif mode == "production":
            config = ConfigPresets.production_backtest()
        else:
            config = TradingConfig()

        # 2) Adjust InfoBus level
        if info_bus_level == "auto":
            if config.debug:
                config.info_bus_audit_level = "DEBUG"
            elif config.live_mode:
                config.info_bus_audit_level = "WARNING"
            else:
                config.info_bus_audit_level = "INFO"
        else:
            config.info_bus_audit_level = str(info_bus_level).upper()

        # 3) Risk level (affects only fallback guard-rails)
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

        # 4) Apply overrides via helper (known fields only)
        if overrides:
            known: Dict[str, Any] = {}
            unknown: List[str] = []

            for key, value in overrides.items():
                if hasattr(config, key):
                    known[key] = value
                else:
                    unknown.append(key)

            if unknown:
                # Simple warning; replace with logger if needed
                print(f"Warning: Unknown config parameters: {unknown}")

            if known:
                # This re-runs __post_init__ without re-loading risk_policy.yaml
                config.apply_overrides(**known)

        # 5) Explicitly set trading mode after config is fully built
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
        print(
            f"  Risk Fallbacks: DD={config.max_drawdown:.1%}, "
            f"Exposure={config.max_total_exposure:.1%}, Pos={config.max_position_pct:.1%}"
        )
        print(f"  Obs Size: {config.environment_observation_size}")
        warns = validate_config(config)
        if warns:
            print(f"  ⚠️ Warnings: {len(warns)}")
            for warning in warns[:3]:
                print(f"    - {warning}")

    print("\n✅ BUS-FIRST configuration system test completed!")
