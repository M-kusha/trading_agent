# -------------------------------------------------------------
# File: modules/position/smart_position_manager.py
# Smart Position Management - Professional Net Position System
#
# Design Principles:
#   1. ONE position per symbol (net position approach)
#   2. No hedging (BUY+SELL same symbol = waste of spread)
#   3. Signal reversal = CLOSE existing, then open new direction
#   4. Smart scaling: add to winners, cut losers early (optional)
#   5. Time-aware exits: aging positions get scrutinized
#
# EXIT LOGIC: Uses unified ExitStrategyEngine (single source of truth)
# This ensures consistency with PositionManager (training) and eliminates
# duplicate exit code. See exit_engine.py for all exit strategy logic.
# -------------------------------------------------------------

from __future__ import annotations

import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import yaml

from modules.utils.audit_utils import RotatingLogger, format_operator_message

# UNIFIED EXIT LOGIC - Single source of truth for all exit decisions
from .exit_engine import (
    ExitStrategyEngine,
    ExitDecision,
    ExitReason,
    PositionContext,
    get_exit_engine,
)


# =========================================================
# YAML CONFIG LOADING
# =========================================================

def load_config_from_yaml() -> Dict[str, Any]:
    """Load smart position config block from risk_policy.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            return policy.get("smart_position", {}) or {}
    except Exception as e:
        print(f"[SmartPositionManager] Failed to load config: {e}")
    return {}


def load_lot_config_from_yaml() -> Dict[str, Any]:
    """Load unified lot sizing config from risk_policy.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            return policy.get("lot_sizing", {}) or {}
    except Exception as e:
        print(f"[SmartPositionManager] Failed to load lot config: {e}")
    return {}


# =========================================================
# CORE TYPES
# =========================================================

class PositionAction(Enum):
    """Clean action types for position management."""
    HOLD = "HOLD"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE = "CLOSE"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    REVERSE = "REVERSE"  # Close + Open opposite
    # Position management actions (new)
    ADJUST_SL = "ADJUST_SL"
    ADJUST_TP = "ADJUST_TP"
    TIGHTEN_PROTECTION = "TIGHTEN_PROTECTION"  # Move SL to breakeven or better


@dataclass
class ExpertSignal:
    """Rich signal data from an expert for position management."""
    expert_name: str
    action: str  # BUY, SELL, HOLD
    confidence: float
    reasoning: str = ""
    supports_position: bool = True  # Does this expert support holding the position?

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expert_name": self.expert_name,
            "action": self.action,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "supports_position": self.supports_position,
        }


@dataclass
class PositionManagementSignal:
    """Full signal context for position management decisions."""
    symbol: str
    expert_signals: List[ExpertSignal] = field(default_factory=list)
    consensus_action: str = "HOLD"  # BUY, SELL, HOLD
    consensus_confidence: float = 0.5
    consensus_score: float = 0.5  # Agreement level among experts
    fragility: float = 0.5  # How fragile is the consensus
    regime: str = "unknown"
    volatility_level: str = "medium"
    # Position-specific context
    position_side: int = 0  # 1=LONG, -1=SHORT, 0=FLAT
    position_pnl: float = 0.0
    position_age_hours: float = 0.0

    @property
    def experts_supporting_position(self) -> int:
        """Count experts that support holding the current position."""
        return sum(1 for e in self.expert_signals if e.supports_position)

    @property
    def experts_against_position(self) -> int:
        """Count experts that recommend closing/reversing."""
        return sum(1 for e in self.expert_signals if not e.supports_position)

    @property
    def support_ratio(self) -> float:
        """Ratio of experts supporting vs total (0-1)."""
        total = len(self.expert_signals)
        return self.experts_supporting_position / total if total > 0 else 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "expert_signals": [e.to_dict() for e in self.expert_signals],
            "consensus_action": self.consensus_action,
            "consensus_confidence": self.consensus_confidence,
            "consensus_score": self.consensus_score,
            "fragility": self.fragility,
            "regime": self.regime,
            "volatility_level": self.volatility_level,
            "position_side": self.position_side,
            "position_pnl": self.position_pnl,
            "position_age_hours": self.position_age_hours,
            "experts_supporting": self.experts_supporting_position,
            "experts_against": self.experts_against_position,
            "support_ratio": self.support_ratio,
        }


@dataclass
class PositionFocusContext:
    """
    Position Focus Context - Tells all modules what position(s) to focus on.

    When active, this signals to all experts and the PPO that they should:
    1. STOP looking for new trade opportunities
    2. EVALUATE whether the current position should be held, scaled, or exited
    3. ASSESS signals in terms of "does this support or threaten our position"

    Published to SmartInfoBus as "position_focus_context" by Executor.
    """
    # Mode flag
    focus_mode_active: bool = False

    # Per-instrument position details
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Primary position (if we want to focus on one)
    primary_instrument: Optional[str] = None
    primary_side: int = 0  # 1=LONG, -1=SHORT
    primary_entry_price: float = 0.0
    primary_current_price: float = 0.0
    primary_pnl: float = 0.0
    primary_pnl_pct: float = 0.0
    primary_lots: float = 0.0
    primary_age_hours: float = 0.0
    primary_ticket: int = 0
    primary_sl: float = 0.0
    primary_tp: float = 0.0

    # Risk context
    total_exposure: float = 0.0
    risk_level: str = "normal"  # normal, elevated, critical

    # Instructions for experts
    evaluation_mode: str = "position_management"  # "new_signals" | "position_management"

    def has_position(self, instrument: str) -> bool:
        """Check if we have a position in the given instrument."""
        return instrument in self.positions

    def get_position_side(self, instrument: str) -> int:
        """Get the side of our position (1=LONG, -1=SHORT, 0=NONE)."""
        pos = self.positions.get(instrument)
        if not pos:
            return 0
        return int(pos.get("side", 0))

    def should_experts_focus_on_position(self) -> bool:
        """Check if experts should switch to position management mode."""
        return self.focus_mode_active and len(self.positions) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "focus_mode_active": self.focus_mode_active,
            "positions": self.positions,
            "primary_instrument": self.primary_instrument,
            "primary_side": self.primary_side,
            "primary_entry_price": self.primary_entry_price,
            "primary_current_price": self.primary_current_price,
            "primary_pnl": self.primary_pnl,
            "primary_pnl_pct": self.primary_pnl_pct,
            "primary_lots": self.primary_lots,
            "primary_age_hours": self.primary_age_hours,
            "primary_ticket": self.primary_ticket,
            "primary_sl": self.primary_sl,
            "primary_tp": self.primary_tp,
            "total_exposure": self.total_exposure,
            "risk_level": self.risk_level,
            "evaluation_mode": self.evaluation_mode,
        }

    @classmethod
    def from_positions(cls, positions: Dict[str, Dict[str, Any]]) -> "PositionFocusContext":
        """Create a PositionFocusContext from a positions dict."""
        if not positions:
            return cls(focus_mode_active=False)

        # Find the primary position (largest notional or first one)
        primary_inst = None
        primary_notional = 0.0
        for inst, pos in positions.items():
            notional = abs(
                float(
                    pos.get("notional_eur", 0.0)
                    or pos.get("units", 0.0) * pos.get("entry_price", 1.0)
                )
            )
            if notional > primary_notional:
                primary_notional = notional
                primary_inst = inst

        ctx = cls(
            focus_mode_active=True,
            positions=positions,
            primary_instrument=primary_inst,
            evaluation_mode="position_management",
        )

        # Fill in primary position details
        if primary_inst and primary_inst in positions:
            pos = positions[primary_inst]
            ctx.primary_side = int(pos.get("side", 0))
            ctx.primary_entry_price = float(pos.get("entry_price", 0.0))
            ctx.primary_current_price = float(
                pos.get("current_price", pos.get("entry_price", 0.0))
            )
            ctx.primary_pnl = float(pos.get("unrealized_pnl", 0.0))
            ctx.primary_lots = float(pos.get("units", pos.get("lots", 0.0)))
            ctx.primary_ticket = int(pos.get("ticket", 0))
            ctx.primary_sl = float(pos.get("sl", 0.0))
            ctx.primary_tp = float(pos.get("tp", 0.0))

            # Calculate PnL percentage
            if ctx.primary_entry_price > 0 and ctx.primary_lots > 0:
                price_diff = ctx.primary_current_price - ctx.primary_entry_price
                if ctx.primary_side < 0:
                    price_diff = -price_diff
                ctx.primary_pnl_pct = (price_diff / ctx.primary_entry_price) * 100

            # Calculate age
            open_time = pos.get("open_time", 0)
            if open_time:
                import time as _time
                from datetime import datetime
                # Handle both Unix timestamp and ISO datetime string formats
                if isinstance(open_time, str):
                    try:
                        # Parse ISO format: '2025-12-09T23:18:53Z'
                        dt = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
                        open_timestamp = dt.timestamp()
                    except (ValueError, AttributeError):
                        open_timestamp = 0.0
                else:
                    open_timestamp = float(open_time)
                
                if open_timestamp > 0:
                    ctx.primary_age_hours = (_time.time() - open_timestamp) / 3600.0

        # Calculate total exposure
        ctx.total_exposure = sum(
            abs(
                float(
                    p.get("notional_eur", 0.0)
                    or float(p.get("units", 0.0)) * float(p.get("entry_price", 1.0))
                )
            )
            for p in positions.values()
        )

        return ctx


@dataclass
class LivePosition:
    """Represents a live position from MT5."""
    symbol: str
    side: int  # 1 = BUY, -1 = SELL
    lots: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    open_time: float  # Unix timestamp
    ticket: int = 0
    sl: float = 0.0
    tp: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.open_time if self.open_time > 0 else 0

    @property
    def age_hours(self) -> float:
        return self.age_seconds / 3600

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0

    @property
    def direction(self) -> str:
        return "BUY" if self.side > 0 else "SELL"


@dataclass
class PPODecisionView:
    """
    Normalised view of Arbiter → PPO decision for one symbol.

    This is what SmartPositionManager cares about:
    - direction/confidence for analytics
    - explicit CLOSE / REVERSE flags for exits
    - whether this is a reversal vs the *current* position

    Version: v5.2 - Supports PPO explicit close/reverse intents from ArbiterLogic
    """
    direction: str = "flat"          # "long" / "short" / "flat"
    confidence: float = 0.0          # 0–1 normalised
    position_size: float = 0.0       # 0–1 relative sizing from Arbiter
    action_intent: str = ""          # "open_long", "close", "reverse", ...

    explicit_close: bool = False     # PPO wants this instrument FLAT
    explicit_reverse: bool = False   # PPO wants hard reverse (close+reopen)

    wants_flat: bool = False         # "target state = no position"
    is_reversal: bool = False        # opposite of current position_side

    # Legacy compatibility - integer direction (-1, 0, 1)
    @property
    def direction_int(self) -> int:
        """Integer direction for backward compatibility: 1=long, -1=short, 0=flat."""
        if self.direction == "long":
            return 1
        elif self.direction == "short":
            return -1
        return 0


@dataclass
class SmartDecision:
    """Result of smart position analysis."""
    action: PositionAction
    symbol: str
    lots: float = 0.0
    side: int = 0  # 1 = BUY, -1 = SELL
    confidence: float = 0.5
    reasons: List[str] = field(default_factory=list)
    close_first: bool = False  # For REVERSE: close before opening
    # Position management fields (new)
    new_sl: Optional[float] = None  # Suggested new stop loss price
    new_tp: Optional[float] = None  # Suggested new take profit price
    expert_support_ratio: float = 0.5  # How many experts support this decision
    management_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "lots": self.lots,
            "side": self.side,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "close_first": self.close_first,
            "new_sl": self.new_sl,
            "new_tp": self.new_tp,
            "expert_support_ratio": self.expert_support_ratio,
            "management_context": self.management_context,
        }


@dataclass
class SmartPositionConfig:
    """
    Configuration for smart position management (v4.0).

    Exit thresholds are managed by ExitStrategyEngine (exit_engine.py).
    Lot sizing uses UnifiedLotCalculator as the single source of truth.
    """
    # Position limits
    max_positions_per_symbol: int = 1
    max_total_positions: int = 4

    # Lot sizing - fallbacks only, UnifiedLotCalculator determines actual size
    default_lot_size: float = 0.01
    max_lot_size: float = 50.0
    use_unified_calculator: bool = True

    # Scaling rules (EUR-based fallback when R-based disabled)
    scale_up_min_profit_eur: float = 50.0
    scale_up_cooldown_seconds: float = 600.0
    scale_down_trigger_loss_eur: float = 40.0
    
    # Scale-up toggle - Creates additional ticket in MT5 hedging mode.
    # SmartPositionManager nets all tickets into 1 logical position per symbol.
    enable_scale_up: bool = True
    
    # Breakeven SL threshold - move SL to entry price when profit reaches this
    # Should match or be close to ExitEngine trailing_activation_eur (€150)
    breakeven_activation_eur: float = 150.0

    # Trade cooldowns
    same_direction_cooldown_seconds: float = 300.0
    reversal_cooldown_seconds: float = 300.0

    # Signal thresholds - kept low, experts already filter weak signals
    min_signal_strength: float = 0.10
    strong_signal_threshold: float = 0.50
    reversal_signal_threshold: float = 0.45

    # PPO exit/reversal thresholds (v5.2+)
    # ppo_exit_conf_threshold: Minimum confidence for PPO explicit close/reversal
    # ppo_reversal_conf_threshold: Legacy threshold (use ppo_exit_conf_threshold)
    ppo_exit_conf_threshold: float = 0.60
    ppo_reversal_conf_threshold: float = 0.60  # Legacy alias

    # Smart scale-down toggle (for losers). Default OFF for safety.
    enable_smart_scale_down: bool = False

    # ═══════════════════════════════════════════════════════════════════════════
    # R-BASED SCALING (v4.0) - Normalize decisions by initial risk
    # ═══════════════════════════════════════════════════════════════════════════
    use_r_based_scaling: bool = True
    scale_up_min_r: float = 1.0              # Scale up after 1R profit
    scale_down_trigger_r: float = -0.5       # Start scale down at 0.5R loss
    scale_down_aggressive_r: float = -1.0    # Aggressive scale down at 1R loss
    scale_down_emergency_r: float = -1.5     # Emergency scale down at 1.5R loss

    # ═══════════════════════════════════════════════════════════════════════════
    # POSITION LIFECYCLE (v4.0) - Stage-aware management
    # ═══════════════════════════════════════════════════════════════════════════
    probe_max_age_hours: float = 0.5         # First 30 min = PROBE
    probe_max_profit_r: float = 0.3          # Exit PROBE if gains 0.3R
    build_max_age_hours: float = 2.0         # Hours 0.5-2 = BUILD
    build_max_profit_r: float = 1.0          # Exit BUILD if gains 1R
    ride_min_profit_r: float = 0.5           # Need 0.5R profit to enter RIDE
    # DEFEND triggers at 20% retrace (BEFORE ExitEngine's 30% trailing close)
    defend_trigger_drawdown_pct: float = 0.20 # Enter DEFEND if profit drops 20%

    # ═══════════════════════════════════════════════════════════════════════════
    # DRAWDOWN-AWARE AGGRESSION (v4.0)
    # ═══════════════════════════════════════════════════════════════════════════
    dd_aggression_enabled: bool = True
    mild_dd_threshold: float = 0.015         # 1.5% DD = mild caution
    moderate_dd_threshold: float = 0.025     # 2.5% DD = moderate caution
    severe_dd_threshold: float = 0.035       # 3.5% DD = defensive mode
    mild_aggression_mult: float = 0.85       # 85% aggression
    moderate_aggression_mult: float = 0.65   # 65% aggression
    severe_aggression_mult: float = 0.40     # 40% aggression

    # ═══════════════════════════════════════════════════════════════════════════
    # STARTUP GRACE (v4.0) - Per-symbol grace period
    # ═══════════════════════════════════════════════════════════════════════════
    default_startup_grace_calls: int = 3     # Default grace for unknown symbols


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION LIFECYCLE STATE MACHINE (v4.0)
# ═══════════════════════════════════════════════════════════════════════════════

class PositionLifecycle(Enum):
    """
    Position lifecycle stages with different management rules.

    PROBE  - First 30 min, tight management, quick exit on weakness
    BUILD  - Hours 0.5-2, allow position to develop, moderate management
    RIDE   - In profit, let it run with trailing protection
    DEFEND - Profit dropping from peak, tighten protection
    EXIT   - Marked for exit (set by ExitEngine)
    """
    PROBE = "probe"
    BUILD = "build"
    RIDE = "ride"
    DEFEND = "defend"
    EXIT = "exit"


# =========================================================
# MAIN MANAGER
# =========================================================

class SmartPositionManager:
    """
    Professional net position management system (v4.0).

    Responsibilities:
      1. Sync with MT5 to know actual positions.
      2. Prevent duplicate/hedge positions.
      3. Smart entry/exit decisions based on signals + P&L.
      4. Time-aware position management with lifecycle states.
      5. Use UnifiedLotCalculator for all lot sizing decisions.
      6. Delegate **all** exit rules to ExitStrategyEngine (single source of truth).
      7. Per-instrument configuration with volatility-aware thresholds.
      8. R-based scaling for professional risk management.
      9. Drawdown-aware aggression control.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Start from sane defaults
        self.config = SmartPositionConfig()

        # Load from YAML if no external config provided
        if config is None:
            config = load_config_from_yaml()

        # Shallow-merge scalar keys from YAML smart_position into dataclass
        # Skip nested dicts like lifecycle / drawdown_aggression / per_instrument
        if config:
            for key, value in config.items():
                if isinstance(value, dict):
                    continue  # handled separately
                if hasattr(self.config, key):
                    setattr(self.config, key, value)

        # Load unified lot sizing config
        lot_config = load_lot_config_from_yaml()
        if lot_config:
            # Override local config with unified values when present
            if "max_lot" in lot_config:
                self.config.max_lot_size = float(lot_config["max_lot"])

        self.logger = RotatingLogger(
            "SmartPositionManager",
            log_path="logs/position/smart_manager.log",
            operator_mode=True,
            max_lines=5000,
        )

        # InfoBus for state persistence (used for profit peak & risk persistence)
        try:
            from modules.utils.info_bus import InfoBusManager
            self._smart_bus = InfoBusManager.get_instance()
        except Exception:
            self._smart_bus = None

        # Initialize unified lot calculator (singleton)
        self._lot_calculator = None

        # STATE TRACKING (v4.0)
        self._positions: Dict[str, LivePosition] = {}       # Net positions per symbol
        self._actual_mt5_position_count: int = 0            # Actual count of MT5 tickets
        self._profit_peaks: Dict[str, float] = self._load_peaks_from_bus()
        self._last_trade_time: Dict[str, float] = {}        # Per-symbol cooldown
        self._last_scale_time: Dict[str, float] = {}        # Per-symbol scale cooldown
        self._last_sync_time: float = 0

        # Per-symbol startup grace (instead of global)
        self._decide_call_count_by_symbol: Dict[str, int] = {}
        self._decide_call_count: int = 0                   # Global fallback (deprecated)
        self._startup_grace_calls: int = 3                 # Deprecated: use _get_symbol_cfg()

        # R-based scaling initial risk
        self._initial_risk_by_symbol: Dict[str, float] = self._load_initial_risk_from_bus()

        # Position lifecycle states (optional sticky override; normally computed on the fly)
        self._lifecycle_states: Dict[str, PositionLifecycle] = {}

        # PER-INSTRUMENT CONFIGURATION (v4.0)
        self._per_instrument_cfg: Dict[str, Dict[str, Any]] = self._load_per_instrument_config()

    # =========================================================
    # CONFIG HELPERS
    # =========================================================

    def _load_per_instrument_config(self) -> Dict[str, Dict[str, Any]]:
        """Load per-instrument configuration and lifecycle/DD overrides from YAML."""
        try:
            import os

            yaml_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config",
                "risk_policy.yaml",
            )
            if not os.path.exists(yaml_path):
                return {}

            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            smart_pos = data.get("smart_position", {}) or {}
            per_inst = smart_pos.get("per_instrument", {}) or {}

            # Lifecycle overrides (flattened onto config)
            lifecycle = smart_pos.get("lifecycle", {}) or {}
            for key, val in lifecycle.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)

            # Drawdown aggression overrides
            dd_agg = smart_pos.get("drawdown_aggression", {}) or {}
            if dd_agg:
                if dd_agg.get("enabled") is not None:
                    self.config.dd_aggression_enabled = bool(dd_agg["enabled"])
                for key in [
                    "mild_dd_threshold",
                    "moderate_dd_threshold",
                    "severe_dd_threshold",
                    "mild_aggression_mult",
                    "moderate_aggression_mult",
                    "severe_aggression_mult",
                ]:
                    if key in dd_agg and hasattr(self.config, key):
                        setattr(self.config, key, dd_agg[key])

            # R-based scaling toggles/thresholds
            r_based = {
                "use_r_based_scaling": smart_pos.get("use_r_based_scaling", True),
                "scale_up_min_r": smart_pos.get("scale_up_min_r", 1.0),
                "scale_down_trigger_r": smart_pos.get("scale_down_trigger_r", -0.5),
                "scale_down_aggressive_r": smart_pos.get("scale_down_aggressive_r", -1.0),
                "scale_down_emergency_r": smart_pos.get("scale_down_emergency_r", -1.5),
            }
            for key, val in r_based.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, val)

            # Optional switch to enable smart scale-down from YAML
            if "enable_smart_scale_down" in smart_pos:
                self.config.enable_smart_scale_down = bool(
                    smart_pos.get("enable_smart_scale_down", False)
                )

            # Scale-up toggle (default OFF - MT5 creates new tickets)
            if "enable_scale_up" in smart_pos:
                self.config.enable_scale_up = bool(
                    smart_pos.get("enable_scale_up", False)
                )

            return per_inst

        except Exception as e:
            self.logger.warning(f"Failed to load per-instrument config: {e}")
            return {}

    # =========================================================
    # SEASONALITY / TRADING WINDOW CHECKS
    # =========================================================

    def _check_no_new_trades(self) -> Tuple[bool, str]:
        """
        Check if SeasonalityRiskExpert says no_new_trades.
        
        Returns:
            Tuple of (no_new_trades, reason)
        """
        try:
            if self._smart_bus:
                seasonality = self._smart_bus.get(
                    "SeasonalityRiskExpert_voting_proposal", "SmartPositionManager"
                )
                if isinstance(seasonality, dict):
                    trading_window = seasonality.get("trading_window", {})
                    if isinstance(trading_window, dict):
                        no_new = trading_window.get("no_new_trades", False)
                        if no_new:
                            local_time = trading_window.get("local_time", "unknown")
                            minutes_to_close = trading_window.get("minutes_to_close", 0)
                            return True, f"Outside trading window (local={local_time}, close_in={minutes_to_close}min)"
        except Exception:
            pass
        return False, ""

    def _check_final_exit_window(self) -> Tuple[bool, bool, float]:
        """
        Check if we're in the final exit window before market close.
        
        In the final exit window (last 60 min before hard_close_hour):
        - No new trades allowed
        - Positions should be exited unless loss is too large
        
        Returns:
            Tuple of (in_final_exit, is_obligatory, max_loss_pct)
        """
        try:
            if self._smart_bus:
                seasonality = self._smart_bus.get(
                    "SeasonalityRiskExpert_voting_proposal", "SmartPositionManager"
                )
                if isinstance(seasonality, dict):
                    trading_window = seasonality.get("trading_window", {})
                    if isinstance(trading_window, dict):
                        final_exit = trading_window.get("final_exit_window", False)
                        obligatory = trading_window.get("final_exit_obligatory", True)
                        max_loss_pct = trading_window.get("final_exit_max_loss_pct", 0.02)
                        return bool(final_exit), bool(obligatory), float(max_loss_pct)
        except Exception:
            pass
        return False, True, 0.02

    def _get_current_balance(self) -> float:
        """Get current account balance from InfoBus for loss % calculation."""
        try:
            if self._smart_bus:
                live_status = self._smart_bus.get(
                    "live_adapter_status", "SmartPositionManager", default=None
                )
                if isinstance(live_status, dict):
                    for key in ("equity", "balance"):
                        val = live_status.get(key)
                        if isinstance(val, (int, float)) and val > 0:
                            return float(val)
        except Exception:
            pass
        return 100_000.0  # Default fallback

    def _get_symbol_cfg(self, symbol: str) -> Dict[str, Any]:
        """
        Get configuration for a specific symbol.

        Returns per-instrument overrides merged with global defaults.
        This is the SINGLE SOURCE OF TRUTH for symbol-specific thresholds.
        """
        cfg: Dict[str, Any] = {
            # P&L / scaling thresholds
            "scale_up_min_profit_eur": self.config.scale_up_min_profit_eur,
            "scale_down_trigger_loss_eur": self.config.scale_down_trigger_loss_eur,
            "min_signal_strength": self.config.min_signal_strength,
            "strong_signal_threshold": self.config.strong_signal_threshold,
            "startup_grace_calls": self.config.default_startup_grace_calls,
            # R-based scaling
            "scale_up_min_r": self.config.scale_up_min_r,
            "scale_down_trigger_r": self.config.scale_down_trigger_r,
            "scale_down_aggressive_r": self.config.scale_down_aggressive_r,
            "scale_down_emergency_r": self.config.scale_down_emergency_r,
            # Lifecycle thresholds (age / regime)
            "probe_max_age_hours": self.config.probe_max_age_hours,
            "probe_max_profit_r": self.config.probe_max_profit_r,
            "build_max_age_hours": self.config.build_max_age_hours,
            "build_max_profit_r": self.config.build_max_profit_r,
            "ride_min_profit_r": self.config.ride_min_profit_r,
            "defend_trigger_drawdown_pct": self.config.defend_trigger_drawdown_pct,
        }

        per_inst = self._per_instrument_cfg.get(symbol)
        if isinstance(per_inst, dict):
            cfg.update(per_inst)

        return cfg


    # =========================================================
    # STATE PERSISTENCE (PEAKS & INITIAL RISK)
    # =========================================================

    def _load_peaks_from_bus(self) -> Dict[str, float]:
        """Load persisted profit peaks from InfoBus (survives restarts)."""
        try:
            if self._smart_bus:
                peaks = self._smart_bus.get("smart_position_peaks", "SmartPositionManager")
                if isinstance(peaks, dict):
                    return {k: float(v) for k, v in peaks.items()}
        except Exception:
            pass
        return {}

    def _persist_peaks_to_bus(self) -> None:
        """Persist profit peaks to InfoBus for restart survival."""
        try:
            if self._smart_bus and self._profit_peaks:
                self._smart_bus.set(
                    "smart_position_peaks",
                    self._profit_peaks.copy(),
                    module="SmartPositionManager",
                    thesis="Persisted peak PnL for trailing stop logic (via ExitEngine)",
                )
        except Exception:
            pass

    def _load_initial_risk_from_bus(self) -> Dict[str, float]:
        """Load persisted initial risk amounts from InfoBus (survives restarts)."""
        try:
            if self._smart_bus:
                risks = self._smart_bus.get(
                    "smart_position_initial_risk", "SmartPositionManager"
                )
                if isinstance(risks, dict):
                    return {k: float(v) for k, v in risks.items()}
        except Exception:
            pass
        return {}

    def _persist_initial_risk_to_bus(self) -> None:
        """Persist initial risk amounts to InfoBus for restart survival."""
        try:
            if self._smart_bus and self._initial_risk_by_symbol:
                self._smart_bus.set(
                    "smart_position_initial_risk",
                    self._initial_risk_by_symbol.copy(),
                    module="SmartPositionManager",
                    thesis="Persisted initial risk for R-based scaling",
                )
        except Exception:
            pass

    def _get_r_multiple(self, symbol: str, pnl: float) -> float:
        """
        Calculate R-multiple for a position.

        R = current_pnl / initial_risk
        R=1 means you've gained what you risked, R=-1 means you've lost what you risked.
        """
        initial_risk = self._initial_risk_by_symbol.get(symbol, 0.0)
        if initial_risk <= 0:
            # Fallback: estimate from global config
            initial_risk = self.config.scale_down_trigger_loss_eur * 2.0
        return pnl / initial_risk if initial_risk > 0 else 0.0

    def _set_initial_risk(self, symbol: str, risk_eur: float) -> None:
        """Record initial risk for a new position (for R-based scaling)."""
        if risk_eur > 0:
            self._initial_risk_by_symbol[symbol] = float(risk_eur)
            self._persist_initial_risk_to_bus()

    # =========================================================
    # LIFECYCLE & AGGRESSION
    # =========================================================

    def _get_lifecycle_state(self, position: LivePosition) -> PositionLifecycle:
        """
        Determine the current lifecycle state of a position.
        """
        symbol = position.symbol
        cfg = self._get_symbol_cfg(symbol)

        # Sticky EXIT override (not currently used, but safe)
        if self._lifecycle_states.get(symbol) == PositionLifecycle.EXIT:
            return PositionLifecycle.EXIT

        # R-based transitions
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
        peak_pnl = self._profit_peaks.get(symbol, position.unrealized_pnl)

        # PROBE
        if position.age_hours < cfg.get("probe_max_age_hours", 0.5):
            if r_mult < cfg.get("probe_max_profit_r", 0.3):
                return PositionLifecycle.PROBE

        # BUILD
        if position.age_hours < cfg.get("build_max_age_hours", 2.0):
            if r_mult < cfg.get("build_max_profit_r", 1.0):
                return PositionLifecycle.BUILD

        # DEFEND – profit retraced from peak (per-instrument override allowed)
        if peak_pnl > 0 and position.unrealized_pnl > 0:
            drawdown_from_peak = (
                (peak_pnl - position.unrealized_pnl) / peak_pnl if peak_pnl > 0 else 0.0
            )
            defend_trigger = cfg.get(
                "defend_trigger_drawdown_pct",
                self.config.defend_trigger_drawdown_pct,
            )
            if drawdown_from_peak >= defend_trigger:
                return PositionLifecycle.DEFEND

        # RIDE – in solid profit (per-instrument override allowed)
        ride_min_r = cfg.get("ride_min_profit_r", self.config.ride_min_profit_r)
        if r_mult >= ride_min_r:
            return PositionLifecycle.RIDE

        # Default
        return PositionLifecycle.BUILD


    def _get_aggression_multiplier(self) -> float:
        """
        Get aggression multiplier based on current drawdown.

        Returns 1.0 for normal, <1.0 during drawdown periods.
        Affects scale up decisions and position sizing.
        """
        if not self.config.dd_aggression_enabled:
            return 1.0

        try:
            if self._smart_bus:
                daily_dd = self._smart_bus.get(
                    "daily_drawdown_pct", "SmartPositionManager", default=0.0
                )
                equity_dd = self._smart_bus.get(
                    "equity_drawdown_pct", "SmartPositionManager", default=0.0
                )
                current_dd = max(float(daily_dd or 0.0), float(equity_dd or 0.0))
            else:
                current_dd = 0.0
        except Exception:
            current_dd = 0.0

        if current_dd >= self.config.severe_dd_threshold:
            return self.config.severe_aggression_mult
        elif current_dd >= self.config.moderate_dd_threshold:
            return self.config.moderate_aggression_mult
        elif current_dd >= self.config.mild_dd_threshold:
            return self.config.mild_aggression_mult

        return 1.0

    # =========================================================
    # STARTUP GRACE
    # =========================================================

    def _get_symbol_call_count(self, symbol: str) -> int:
        """Get the number of decide() calls for a specific symbol."""
        return self._decide_call_count_by_symbol.get(symbol, 0)

    def _increment_symbol_call_count(self, symbol: str) -> None:
        """Increment decide() call count for a symbol."""
        self._decide_call_count_by_symbol[symbol] = (
            self._get_symbol_call_count(symbol) + 1
        )
        self._decide_call_count += 1

    def _is_signal_valid_for_symbol(self, symbol: str) -> bool:
        """Check if we've had enough calls to trust signals for this symbol."""
        cfg = self._get_symbol_cfg(symbol)
        grace_calls = cfg.get(
            "startup_grace_calls", self.config.default_startup_grace_calls
        )
        return self._get_symbol_call_count(symbol) >= grace_calls

    # =========================================================
    # PPO DECISION AWARENESS (v5.2+)
    # PPO is the MASTER for all trading decisions - experts are advisory only
    # Now supports explicit close/reverse intents from ArbiterLogic
    # =========================================================

    def _get_ppo_decision(
        self,
        symbol: str,
        position_side: Optional[str] = None,
    ) -> PPODecisionView:
        """
        Read Arbiter → PPO multi-instrument decision from SmartInfoBus
        and normalise it for position management.

        - Uses `ppo_multi_decision` (ArbiterMultiDecision → dict) as primary source
        - Falls back to legacy `ppo_decision` if needed
        - Understands:
          * meta.action_intent
          * meta.ppo_exit.explicit_close / explicit_reverse

        Args:
            symbol: MT5 symbol, e.g. "XAUUSD", "EURUSD"
            position_side: current position side for this symbol ("long"/"short"/None)
                           If not provided, will be inferred from self._positions

        Returns:
            PPODecisionView with all PPO intent information
        """
        view = PPODecisionView()

        if self._smart_bus is None:
            return view

        # Infer position_side if not provided
        if position_side is None:
            position = self._positions.get(symbol)
            if position is not None:
                position_side = "long" if position.side > 0 else "short"

        def _normalise_symbol_key(key: str) -> str:
            return key.replace("_", "").replace(".", "").upper()

        sym_key = _normalise_symbol_key(symbol)

        try:
            # ─────────────────────────────────────────────────────────
            # 1) Preferred: structured multi-instrument decision
            # ─────────────────────────────────────────────────────────
            multi_decision = self._smart_bus.get(
                "ppo_multi_decision",
                "SmartPositionManager",
                default=None,
            )

            inst_dec = None

            # Dataclass style: multi_decision.instruments: Dict[str, InstrumentDecision]
            if hasattr(multi_decision, "instruments"):
                inst_map = getattr(multi_decision, "instruments", {}) or {}
                for k, v in inst_map.items():
                    if _normalise_symbol_key(str(k)) == sym_key:
                        inst_dec = v
                        break

            # Dict style: {"instruments": {...}} or {"decisions": {...}}
            if inst_dec is None and isinstance(multi_decision, dict):
                inst_map = (
                    multi_decision.get("instruments")
                    or multi_decision.get("decisions")
                    or {}
                )
                if isinstance(inst_map, dict):
                    # Try direct, then normalised keys
                    inst_dec = inst_map.get(symbol)
                    if inst_dec is None:
                        for k, v in inst_map.items():
                            if _normalise_symbol_key(str(k)) == sym_key:
                                inst_dec = v
                                break

            if inst_dec is not None:
                # dataclass vs dict normalisation
                if not isinstance(inst_dec, dict):
                    # Assume InstrumentDecision dataclass
                    direction = str(getattr(inst_dec, "direction", "flat") or "flat").lower()
                    confidence = float(getattr(inst_dec, "confidence", 0.0) or 0.0)
                    position_size = float(getattr(inst_dec, "position_size", 0.0) or 0.0)
                    meta = getattr(inst_dec, "meta", {}) or {}
                else:
                    direction = str(inst_dec.get("direction", "flat") or "flat").lower()
                    confidence = float(inst_dec.get("confidence", 0.0) or 0.0)
                    position_size = float(inst_dec.get("position_size", 0.0) or 0.0)
                    meta = inst_dec.get("meta", {}) or {}

                action_intent = str(meta.get("action_intent", "") or "")
                ppo_exit = meta.get("ppo_exit", {}) or {}

                explicit_close = bool(ppo_exit.get("explicit_close", False))
                explicit_reverse = bool(ppo_exit.get("explicit_reverse", False))

                wants_flat = (
                    direction == "flat"
                    or action_intent in ("close", "no_position")
                    or explicit_close
                )

                # Reversal = opposite direction vs current live position
                is_reversal = False
                if position_side in ("long", "short"):
                    if explicit_reverse:
                        is_reversal = True
                    elif direction in ("long", "short"):
                        is_reversal = (
                            position_side == "long" and direction == "short"
                        ) or (
                            position_side == "short" and direction == "long"
                        )

                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.position_size = max(0.0, min(1.0, position_size))
                view.action_intent = action_intent
                view.explicit_close = explicit_close
                view.explicit_reverse = explicit_reverse
                view.wants_flat = wants_flat
                view.is_reversal = is_reversal

                self.logger.debug(
                    f"[PPO_DECISION] {symbol}: dir={view.direction} conf={view.confidence:.3f} "
                    f"size={view.position_size:.3f} intent={view.action_intent or '-'} "
                    f"close={view.explicit_close} reverse={view.explicit_reverse} "
                    f"flat={view.wants_flat} rev={view.is_reversal}"
                )
                return view

            # ─────────────────────────────────────────────────────────
            # 2) Legacy single-instrument decision (older PPO shell)
            # ─────────────────────────────────────────────────────────
            legacy_dec = self._smart_bus.get(
                "ppo_decision",
                "SmartPositionManager",
                default=None,
            )

            if isinstance(legacy_dec, dict):
                direction = str(legacy_dec.get("direction", "flat") or "flat").lower()
                confidence = float(legacy_dec.get("confidence", 0.0) or 0.0)

                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.wants_flat = direction == "flat"

                if position_side in ("long", "short") and direction in ("long", "short"):
                    view.is_reversal = (
                        position_side == "long" and direction == "short"
                    ) or (
                        position_side == "short" and direction == "long"
                    )

                self.logger.debug(
                    f"[PPO_DECISION_LEGACY] {symbol}: dir={view.direction} "
                    f"conf={view.confidence:.3f} rev={view.is_reversal}"
                )
                return view

            # Also try ppo_final_decision as fallback
            final_dec = self._smart_bus.get(
                "ppo_final_decision",
                "SmartPositionManager",
                default=None,
            )

            if isinstance(final_dec, dict):
                direction = str(final_dec.get("direction", "flat") or "flat").lower()
                confidence = float(final_dec.get("confidence", 0.0) or 0.0)

                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.wants_flat = direction == "flat"

                if position_side in ("long", "short") and direction in ("long", "short"):
                    view.is_reversal = (
                        position_side == "long" and direction == "short"
                    ) or (
                        position_side == "short" and direction == "long"
                    )

                self.logger.debug(
                    f"[PPO_DECISION_FINAL] {symbol}: dir={view.direction} "
                    f"conf={view.confidence:.3f} rev={view.is_reversal}"
                )
                return view

        except Exception as e:
            self.logger.warning(
                f"[PPO_DECISION] Failed to read PPO decision for {symbol}: {e}"
            )

        return view

    def _should_respect_ppo_reversal(
        self,
        symbol: str,
        ppo_confidence: float,
        position_pnl: float,
    ) -> tuple[bool, str]:
        """
        Determine if we should respect PPO's reversal signal over expert opinions.
        
        PPO is the MASTER - when PPO signals reversal with high confidence, 
        we should respect it regardless of what experts say.
        
        Args:
            symbol: Trading symbol
            ppo_confidence: PPO's confidence in the reversal (0.0-1.0)
            position_pnl: Current position P&L
            
        Returns:
            tuple of (should_respect, reason)
        """
        # PPO reversal threshold - need strong conviction to override
        # Lower threshold for losing positions (get out faster)
        if position_pnl < 0:
            reversal_threshold = 0.60  # 60% confidence for losers
        else:
            reversal_threshold = 0.70  # 70% confidence for winners
        
        if ppo_confidence >= reversal_threshold:
            reason = (
                f"PPO MASTER REVERSAL: confidence {ppo_confidence:.0%} >= "
                f"{reversal_threshold:.0%} threshold (P&L €{position_pnl:.2f})"
            )
            return (True, reason)
        
        return (False, f"PPO confidence {ppo_confidence:.0%} < {reversal_threshold:.0%}")

    @property
    def lot_calculator(self):
        """Lazy-load the unified lot calculator."""
        if self._lot_calculator is None:
            try:
                from modules.utils.lot_calculator import UnifiedLotCalculator

                self._lot_calculator = UnifiedLotCalculator.get_instance()
            except Exception as e:
                self.logger.warning(f"Failed to load UnifiedLotCalculator: {e}")
        return self._lot_calculator

    # =========================================================
    # POSITION SYNC
    # =========================================================

    def sync_positions(self, mt5_positions: List[Dict[str, Any]]) -> Dict[str, LivePosition]:
        """
        Sync positions from MT5 data.

        Args:
            mt5_positions: List of position dicts from MT5 adapter

        Returns:
            Dict of symbol -> LivePosition (net positions)
        """
        self._positions.clear()

        # Track actual MT5 position count BEFORE netting
        self._actual_mt5_position_count = len(mt5_positions) if mt5_positions else 0

        # Group by symbol and calculate net position
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for pos in mt5_positions:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol.setdefault(symbol, []).append(pos)

        for symbol, positions in by_symbol.items():
            net_pos = self._calculate_net_position(symbol, positions)
            if net_pos and abs(net_pos.lots) > 0.001:
                self._positions[symbol] = net_pos
                prev_peak = self._profit_peaks.get(symbol, net_pos.unrealized_pnl)
                self._profit_peaks[symbol] = max(prev_peak, net_pos.unrealized_pnl)

        # Clean up profit peaks for closed symbols and reset ExitEngine peaks accordingly
        closed_symbols = set(self._profit_peaks.keys()) - set(self._positions.keys())
        if closed_symbols:
            exit_engine = get_exit_engine()
            for symbol in closed_symbols:
                self._profit_peaks.pop(symbol, None)
                try:
                    exit_engine.reset_peak(symbol)
                except Exception:
                    # Safety: exit engine reset must never break sync
                    pass

        self._persist_peaks_to_bus()
        self._last_sync_time = time.time()
        return self._positions.copy()

    def _calculate_net_position(
        self,
        symbol: str,
        positions: List[Dict[str, Any]],
    ) -> Optional[LivePosition]:
        """Calculate net position from potentially multiple tickets."""
        if not positions:
            return None

        buy_lots = sell_lots = 0.0
        buy_value = sell_value = 0.0
        total_pnl = 0.0
        earliest_time = float("inf")
        current_price = 0.0

        largest_buy_ticket = 0
        largest_buy_lots = 0.0
        largest_sell_ticket = 0
        largest_sell_lots = 0.0

        position_sl = 0.0
        position_tp = 0.0

        for pos in positions:
            lots = float(pos.get("volume", pos.get("lots", 0.0)) or 0.0)
            entry = float(pos.get("price_open", pos.get("entry_price", 0.0)) or 0.0)
            pnl = float(pos.get("profit", pos.get("unrealized_pnl", 0.0)) or 0.0)
            open_time = float(pos.get("time", pos.get("open_time", 0.0)) or 0.0)
            pos_type = pos.get("type", 0)
            ticket = int(pos.get("ticket", 0) or 0)
            sl = float(pos.get("sl", 0.0) or 0.0)
            tp = float(pos.get("tp", 0.0) or 0.0)

            is_buy = pos_type == 0  # POSITION_TYPE_BUY = 0

            if is_buy:
                buy_lots += lots
                buy_value += lots * entry
                if lots > largest_buy_lots:
                    largest_buy_lots = lots
                    largest_buy_ticket = ticket
            else:
                sell_lots += lots
                sell_value += lots * entry
                if lots > largest_sell_lots:
                    largest_sell_lots = lots
                    largest_sell_ticket = ticket

            total_pnl += pnl
            if open_time > 0:
                earliest_time = min(earliest_time, open_time)

            if current_price == 0.0:
                current_price = float(pos.get("price_current", entry) or entry)

            # Track SL/TP from position with most lots
            if lots >= max(largest_buy_lots, largest_sell_lots):
                position_sl = sl
                position_tp = tp

        net_lots = buy_lots - sell_lots
        if abs(net_lots) < 0.001:
            return None

        side = 1 if net_lots > 0 else -1
        abs_lots = abs(net_lots)

        # VWAP entry price for net side
        if side > 0 and buy_lots > 0:
            entry_price = buy_value / buy_lots
        elif side < 0 and sell_lots > 0:
            entry_price = sell_value / sell_lots
        else:
            entry_price = current_price

        ticket = largest_buy_ticket if side > 0 else largest_sell_ticket

        return LivePosition(
            symbol=symbol,
            side=side,
            lots=abs_lots,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=total_pnl,
            open_time=earliest_time if earliest_time < float("inf") else time.time(),
            ticket=ticket,
            sl=position_sl,
            tp=position_tp,
        )

    # =========================================================
    # DECISION FABRIC
    # =========================================================

    def _make_decision(
        self,
        action: PositionAction,
        symbol: str,
        lots: float = 0.0,
        side: int = 0,
        confidence: float = 0.5,
        reasons: Optional[List[str]] = None,
        close_first: bool = False,
        new_sl: Optional[float] = None,
        new_tp: Optional[float] = None,
        expert_support_ratio: float = 0.5,
        management_context: Optional[Dict[str, Any]] = None,
    ) -> SmartDecision:
        dec = SmartDecision(
            action=action,
            symbol=symbol,
            lots=lots,
            side=side,
            confidence=confidence,
            reasons=reasons or [],
            close_first=close_first,
            new_sl=new_sl,
            new_tp=new_tp,
            expert_support_ratio=expert_support_ratio,
            management_context=management_context or {},
        )

        # For any explicit CLOSE/REVERSE we also reset state for that symbol
        if dec.action in (PositionAction.CLOSE, PositionAction.REVERSE):
            if dec.symbol in self._profit_peaks:
                self._profit_peaks.pop(dec.symbol, None)
            self._lifecycle_states.pop(dec.symbol, None)
            self._initial_risk_by_symbol.pop(dec.symbol, None)
            self._persist_initial_risk_to_bus()
            try:
                exit_engine = get_exit_engine()
                exit_engine.reset_peak(dec.symbol)
            except Exception:
                pass

        self._log_decision_box(dec)
        return dec

    

    def _log_decision_box(self, decision: SmartDecision) -> None:
        """Log a structured box-style summary for smart decisions."""
        if decision.action == PositionAction.HOLD:
            return
        try:
            header = f"SMART POSITION DECISION – {decision.symbol}"
            box_width = 78

            lines_box: List[str] = []
            lines_box.append("")
            lines_box.append(
                "┌─ " + header + " " + "─" * max(0, box_width - len(header) - 3)
            )
            lines_box.append(
                f"│ Action:        {decision.action.value:<18} Side: {decision.side:+d}"
            )
            lines_box.append(
                f"│ Lots:          {decision.lots:.2f}                "
                f"Confidence: {decision.confidence:.2f}"
            )

            if decision.new_sl is not None or decision.new_tp is not None:
                sl_str = (
                    f"{decision.new_sl:.5f}"
                    if decision.new_sl is not None
                    else "unchanged"
                )
                tp_str = (
                    f"{decision.new_tp:.5f}"
                    if decision.new_tp is not None
                    else "unchanged"
                )
                lines_box.append(
                    f"│ New SL:        {sl_str:<18} New TP: {tp_str}"
                )

            if decision.expert_support_ratio != 0.5:
                lines_box.append(
                    f"│ Expert Support: {decision.expert_support_ratio:.0%}"
                )

            if decision.close_first:
                lines_box.append(
                    "│ Note:          Close existing position before applying action"
                )

            if decision.reasons:
                lines_box.append("│ Reasons:")
                for r in decision.reasons[:3]:
                    lines_box.append(f"│   • {r}")

            lines_box.append("└" + "─" * (box_width - 1))

            self.logger.info("\n".join(lines_box))
        except Exception:
            # Box logging should never affect trading
            pass

    # =========================================================
    # PUBLIC DECISION API - SIMPLE MODE
    # =========================================================

    def decide(
        self,
        symbol: str,
        signal_direction: int,   # 1 = BUY signal, -1 = SELL signal, 0 = neutral
        signal_strength: float,  # 0.0 to 1.0
        consensus_confidence: float = 0.5,
    ) -> SmartDecision:
        """
        Make a smart position decision for a single symbol.
        """
        cfg = self.config

        # Per-instrument config
        sym_cfg = self._get_symbol_cfg(symbol)

        # Per-symbol startup grace
        self._increment_symbol_call_count(symbol)
        signal_valid = self._is_signal_valid_for_symbol(symbol)

        # Sanitise inputs
        if signal_direction > 0:
            signal_direction = 1
        elif signal_direction < 0:
            signal_direction = -1
        else:
            signal_direction = 0

        signal_strength = float(max(0.0, min(1.0, signal_strength)))
        consensus_confidence = float(max(0.0, min(1.0, consensus_confidence)))

        position = self._positions.get(symbol)
        reasons: List[str] = []

        # Case 1: No existing position
        if position is None:
            return self._decide_new_position(
                symbol=symbol,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=consensus_confidence,
                reasons=reasons,
            )

        # Update local profit peak
        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        # ═══════════════════════════════════════════════════════════════════
        # FINAL EXIT WINDOW CHECK (Last 60 min before hard_close_hour)
        # In the final exit window, we push to exit positions to avoid 
        # overnight exposure. However, we only exit if the loss is not
        # too large (configurable max_loss_pct, default 2% of equity).
        # This prevents locking in large losses just before market close.
        # ═══════════════════════════════════════════════════════════════════
        in_final_exit, is_obligatory, max_loss_pct = self._check_final_exit_window()
        if in_final_exit and is_obligatory:
            balance = self._get_current_balance()
            max_loss_amount = balance * max_loss_pct
            
            if position.unrealized_pnl >= 0:
                # Position is profitable or breakeven - exit to avoid overnight
                reasons.append(
                    f"🌙 FINAL EXIT WINDOW: Closing profitable/breakeven position "
                    f"(P&L €{position.unrealized_pnl:.2f}) to avoid overnight exposure"
                )
                self.logger.info(
                    f"[FINAL_EXIT] {symbol}: Closing position in final exit window "
                    f"(P&L €{position.unrealized_pnl:.2f})"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.80,
                    reasons=reasons,
                )
            elif abs(position.unrealized_pnl) <= max_loss_amount:
                # Position has acceptable loss - exit to avoid overnight
                reasons.append(
                    f"🌙 FINAL EXIT WINDOW: Closing position with acceptable loss "
                    f"(P&L €{position.unrealized_pnl:.2f} within €{max_loss_amount:.2f} limit) "
                    f"to avoid overnight exposure"
                )
                self.logger.info(
                    f"[FINAL_EXIT] {symbol}: Closing losing position in final exit window "
                    f"(P&L €{position.unrealized_pnl:.2f} within €{max_loss_amount:.2f} limit)"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                )
            else:
                # Loss is too large - hold and let ExitEngine manage
                reasons.append(
                    f"⚠️ FINAL EXIT WINDOW: Holding position with large loss "
                    f"(P&L €{position.unrealized_pnl:.2f} exceeds €{max_loss_amount:.2f} limit) - "
                    f"letting ExitEngine manage"
                )
                self.logger.warning(
                    f"[FINAL_EXIT] {symbol}: Position loss €{position.unrealized_pnl:.2f} "
                    f"exceeds max €{max_loss_amount:.2f} - not forcing close"
                )

        # Case 2: Existing position – unified ExitEngine first
        lifecycle = self._get_lifecycle_state(position)
        
        exit_ctx = PositionContext(
            symbol=symbol,
            side=position.side,
            unrealized_pnl=position.unrealized_pnl,
            peak_pnl=self._profit_peaks.get(symbol, position.unrealized_pnl),
            entry_price=position.entry_price,
            current_price=position.current_price,
            open_time=position.open_time,
            lots=position.lots,
            position_id=str(position.ticket) if hasattr(position, "ticket") else "",
            signal_direction=signal_direction,
            signal_strength=signal_strength,
            signal_valid=signal_valid,
            consensus_confidence=consensus_confidence,
            # Pass lifecycle as regime - "defend" triggers tighter trailing in ExitEngine
            regime=lifecycle.value if lifecycle else "normal",
        )

        exit_engine = get_exit_engine()
        exit_decision = exit_engine.evaluate(exit_ctx)

        if exit_decision.should_exit:
            reason_msg = exit_decision.details.get("message", exit_decision.reason.name)
            reasons.append(f"[ExitEngine] {reason_msg}")

            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=exit_decision.confidence,
                reasons=reasons,
            )

        # Case 3: Position exists, signal aligns – consider scaling up
        signal_against = (position.side > 0 and signal_direction < 0) or (
            position.side < 0 and signal_direction > 0
        )
        signal_aligns = (position.side > 0 and signal_direction > 0) or (
            position.side < 0 and signal_direction < 0
        )

        if signal_aligns and signal_direction != 0:
            return self._decide_scale(
                symbol=symbol,
                position=position,
                signal_strength=signal_strength,
                consensus_confidence=consensus_confidence,
                reasons=reasons,
            )

        # Case 4: Signal opposes but ExitEngine said HOLD
        # ═══════════════════════════════════════════════════════════════════
        # PPO MASTER EXIT / REVERSAL CHECK (v5.2+)
        # PPO now has explicit close/reverse intents from ArbiterLogic
        # If PPO explicitly wants to close/reverse, respect it over committee
        # ═══════════════════════════════════════════════════════════════════
        side_str = "long" if position.side > 0 else "short"
        ppo_view = self._get_ppo_decision(symbol, side_str)

        # Single configurable threshold for PPO-led exits
        ppo_exit_threshold = getattr(
            self.config,
            "ppo_exit_conf_threshold",
            getattr(self.config, "ppo_reversal_conf_threshold", 0.6),
        )

        # 1) Hard PPO explicit close: PPO explicitly wants this instrument closed
        if ppo_view.explicit_close and ppo_view.confidence >= ppo_exit_threshold:
            reasons.append(
                f"🎯 PPO explicit CLOSE intent (conf={ppo_view.confidence:.2f})"
            )
            self.logger.info(
                f"[PPO_EXIT] {symbol}: PPO explicit CLOSE (conf={ppo_view.confidence:.3f})"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=max(0.75, ppo_view.confidence),
                reasons=reasons,
            )

        # 2) PPO reversal: wants to flip side (close + re-enter opposite)
        if ppo_view.is_reversal and ppo_view.confidence >= ppo_exit_threshold:
            should_respect, ppo_reason = self._should_respect_ppo_reversal(
                symbol, ppo_view.confidence, position.unrealized_pnl
            )
            if should_respect:
                reasons.append(f"🎯 {ppo_reason}")
                signal_against = True  # Force reversal path
                signal_strength = max(signal_strength, ppo_view.confidence)
                self.logger.info(
                    f"[PPO_EXIT] {symbol}: PPO reversal {side_str} → {ppo_view.direction} "
                    f"(conf={ppo_view.confidence:.3f})"
                )

        if signal_against and signal_strength >= cfg.strong_signal_threshold:
            reasons.append(
                f"REVERSAL: Strong opposing signal ({signal_strength:.2f}) "
                f"vs {position.direction} position"
            )

            # If losing, prioritise closing regardless of cooldown
            if position.unrealized_pnl < 0:
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                )

            # If profitable: respect reversal cooldown
            last_trade = self._last_trade_time.get(symbol, 0.0)
            elapsed = time.time() - last_trade
            if elapsed < cfg.reversal_cooldown_seconds:
                remaining = max(0.0, cfg.reversal_cooldown_seconds - elapsed)
                reasons.append(
                    f"Reversal cooldown active ({remaining:.0f}s remaining) – "
                    f"skipping flip, prefer HOLD / manual management"
                )
                return self._make_decision(
                    action=PositionAction.HOLD,
                    symbol=symbol,
                    confidence=0.65,
                    reasons=reasons,
                )

            # No cooldown issue: check trading window before REVERSE
            # REVERSE creates a new position, so it's blocked when no_new_trades=True
            no_new_trades, no_new_reason = self._check_no_new_trades()
            if no_new_trades:
                reasons.append(
                    f"🚫 REVERSE BLOCKED: {no_new_reason} - closing instead"
                )
                # Just close the position, don't reverse
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                )
            
            # Allow REVERSE (Executor must close+open)
            return self._make_decision(
                action=PositionAction.REVERSE,
                symbol=symbol,
                lots=self.config.default_lot_size,
                side=signal_direction,
                confidence=0.70,
                reasons=reasons,
                close_first=True,
            )

        # Case 5: Nothing actionable – HOLD
        reasons.append(
            f"HOLD: {position.direction} {position.lots:.2f} lots, "
            f"P&L €{position.unrealized_pnl:.2f}, age {position.age_hours:.1f}h"
        )
        return self._make_decision(
            action=PositionAction.HOLD,
            symbol=symbol,
            confidence=0.60,
            reasons=reasons,
        )

    # =========================================================
    # NEW POSITION LOGIC
    # =========================================================

    def _decide_new_position(
        self,
        symbol: str,
        signal_direction: int,
        signal_strength: float,
        consensus_confidence: float,
        reasons: List[str],
    ) -> SmartDecision:
        """Decide on opening a new position."""
        cfg = self.config
        sym_cfg = self._get_symbol_cfg(symbol)

        # ═══════════════════════════════════════════════════════════════════
        # TRADING WINDOW CHECK - Block new trades when no_new_trades=True
        # This ensures we don't open positions outside trading hours or
        # in the final exit window before market close.
        # CLOSE/TIGHTEN operations are still allowed for existing positions.
        # ═══════════════════════════════════════════════════════════════════
        no_new_trades, no_new_reason = self._check_no_new_trades()
        if no_new_trades:
            reasons.append(f"🚫 BLOCKED: {no_new_reason}")
            self.logger.info(
                f"[TRADING_WINDOW] {symbol}: Blocked new position - {no_new_reason}"
            )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Race-condition guard: if sync is slightly behind, don't double-open
        existing = self._positions.get(symbol)
        if existing is not None:
            if (existing.side > 0 and signal_direction < 0) or (
                existing.side < 0 and signal_direction > 0
            ):
                reasons.append(
                    f"BLOCKED: Existing {existing.direction} position exists, "
                    f"won't open opposing position"
                )
            else:
                reasons.append(
                    f"BLOCKED: Already have {existing.direction} position, "
                    f"prefer SCALE instead of new OPEN"
                )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Per-instrument min signal strength
        min_strength = sym_cfg.get("min_signal_strength", cfg.min_signal_strength)
        if signal_strength < min_strength:
            reasons.append(
                f"Signal too weak ({signal_strength:.2f} < {min_strength:.2f})"
            )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Neutral signals do nothing
        if signal_direction == 0:
            reasons.append("Neutral signal - no direction")
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Same-direction cooldown
        last_trade = self._last_trade_time.get(symbol, 0.0)
        cooldown_remaining = (
            cfg.same_direction_cooldown_seconds - (time.time() - last_trade)
        )
        if cooldown_remaining > 0:
            reasons.append(f"Cooldown active ({cooldown_remaining:.0f}s remaining)")
            # Publish per-symbol cooldown to SmartInfoBus so PPO/experts can observe it.
            try:
                if self._smart_bus is not None:
                    cooldown_state = self._smart_bus.get(
                        "instrument_cooldown_state",
                        "SmartPositionManager",
                        default={},
                    ) or {}
                    if isinstance(cooldown_state, dict):
                        sym_state = cooldown_state.get(symbol, {})
                        if not isinstance(sym_state, dict):
                            sym_state = {}
                        sym_state.update(
                            {
                                "on_cooldown": True,
                                "cooldown_remaining": float(max(cooldown_remaining, 0.0)),
                                "last_trade_ts": float(last_trade),
                                "same_direction_cooldown_seconds": float(
                                    cfg.same_direction_cooldown_seconds
                                ),
                            }
                        )
                        cooldown_state[symbol] = sym_state
                        self._smart_bus.set(
                            "instrument_cooldown_state",
                            cooldown_state,
                            module="SmartPositionManager",
                            thesis="Per-instrument trade cooldown state",
                        )
            except Exception:
                # Cooldown publishing is advisory only; never block decisions on failure.
                pass
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Total position count (use ACTUAL MT5 count to see hedges)
        effective_count = max(len(self._positions), self._actual_mt5_position_count)
        if effective_count >= cfg.max_total_positions:
            reasons.append(
                "Max positions reached "
                f"(net={len(self._positions)}, actual={self._actual_mt5_position_count}, "
                f"max={cfg.max_total_positions})"
            )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # All checks passed - open position
        action = (
            PositionAction.OPEN_LONG if signal_direction > 0 else PositionAction.OPEN_SHORT
        )

        # UNIFIED LOT CALCULATION
        lots = 0.0
        initial_risk_eur = 0.0
        if self.lot_calculator and self.config.use_unified_calculator:
            try:
                lots, lot_details = self.lot_calculator.calculate_lots(
                    symbol=symbol,
                    signal_strength=signal_strength,
                )
                initial_risk_eur = float(
                    lot_details.get(
                        "risk_eur",
                        lot_details.get("risk_pct", 0.005) * lot_details.get("balance", 100000),
                    )
                )

                self.logger.info(
                    f"[SMART_PM] 📊 UNIFIED_LOT: {symbol} | signal={signal_strength:.2f} | "
                    f"lots={lots:.2f} | balance=€{lot_details.get('balance', 0):.0f} | "
                    f"risk={lot_details.get('risk_pct', 0)*100:.1f}% | "
                    f"initial_risk_eur=€{initial_risk_eur:.2f}"
                )
            except Exception as e:
                self.logger.warning(
                    f"[SMART_PM] Unified lot calc failed: {e}, using fallback"
                )
                lots = 0.0

        # Fallback if unified calculator unavailable or failed
        if lots <= 0:
            raw_lots = cfg.default_lot_size * signal_strength
            lots = min(max(raw_lots, 0.01), cfg.max_lot_size)
            initial_risk_eur = 50.0  # Conservative estimate

        self._set_initial_risk(symbol, initial_risk_eur)
        self._lifecycle_states[symbol] = PositionLifecycle.PROBE

        reasons.append(
            f"OPEN {action.value}: signal={signal_strength:.2f}, "
            f"consensus={consensus_confidence:.2f}, lots={lots:.2f}, "
            f"initial_risk=€{initial_risk_eur:.2f}"
        )

        return self._make_decision(
            action=action,
            symbol=symbol,
            lots=round(lots, 2),
            side=signal_direction,
            confidence=min(signal_strength, consensus_confidence),
            reasons=reasons,
        )

    # =========================================================
    # SCALE LOGIC (ALIGNING SIGNALS)
    # =========================================================

    def _decide_scale(
        self,
        symbol: str,
        position: LivePosition,
        signal_strength: float,
        consensus_confidence: float,
        reasons: List[str],
    ) -> SmartDecision:
        """
        Decide on scaling an existing position.

        Scaling up:
          - Signal aligns
          - Experts confident
          - Position profitable enough (R-based or EUR-based)
          - Cooldown passed
          - NOT in DEFEND state (profit retracing from peak)
          
        NOTE: Scale-up is DISABLED by default (enable_scale_up=false) because
        MT5 creates new tickets instead of modifying existing positions.
        """
        cfg = self.config

        # ═══════════════════════════════════════════════════════════════════
        # SCALE-UP DISABLED CHECK
        # MT5 hedging mode creates new tickets for each order. There's no way
        # to increase lot size on an existing ticket. Keep scale-up disabled
        # to maintain 1 ticket per symbol (prop firm friendly).
        # ═══════════════════════════════════════════════════════════════════
        if not cfg.enable_scale_up:
            # Just hold - don't scale up
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.60,
                reasons=reasons,
            )

        symbol = position.symbol

        # ═══════════════════════════════════════════════════════════════════
        # DEFEND MODE CHECK - Block scale-up when profit is retracing!
        # This is the PRIMARY purpose of DEFEND state.
        # ═══════════════════════════════════════════════════════════════════
        lifecycle = self._get_lifecycle_state(position)
        if lifecycle == PositionLifecycle.DEFEND:
            r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
            peak_pnl = self._profit_peaks.get(symbol, position.unrealized_pnl)
            retrace_pct = (peak_pnl - position.unrealized_pnl) / peak_pnl if peak_pnl > 0 else 0.0
            reasons.append(
                f"🛡️ DEFEND MODE: Profit retracing ({retrace_pct:.0%} from peak €{peak_pnl:.2f}), "
                f"blocking scale-up. Current €{position.unrealized_pnl:.2f} ({r_mult:.1f}R)"
            )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.60,
                reasons=reasons,
            )

        last_scale = self._last_scale_time.get(symbol, 0.0)
        scale_cooldown_ok = (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds

        sym_cfg = self._get_symbol_cfg(symbol)
        strong_threshold = sym_cfg.get("strong_signal_threshold", cfg.strong_signal_threshold)

        use_r_based = self.config.use_r_based_scaling
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
        scale_up_min_r = sym_cfg.get("scale_up_min_r", self.config.scale_up_min_r)
        scale_up_min_profit = sym_cfg.get(
            "scale_up_min_profit_eur", cfg.scale_up_min_profit_eur
        )

        profit_ok = (
            (use_r_based and r_mult >= scale_up_min_r)
            or (not use_r_based and position.unrealized_pnl >= scale_up_min_profit)
        )

        if (
            signal_strength >= strong_threshold
            and profit_ok
            and scale_cooldown_ok
            and position.lots < cfg.max_lot_size
        ):
            add_lots = 0.0
            if self.lot_calculator and self.config.use_unified_calculator:
                try:
                    aggression = self._get_aggression_multiplier()
                    scale_signal = signal_strength * 0.5 * aggression
                    add_lots, _ = self.lot_calculator.calculate_lots(
                        symbol=symbol,
                        signal_strength=scale_signal,
                    )
                    add_lots = min(add_lots, cfg.max_lot_size - position.lots)
                except Exception:
                    add_lots = 0.0

            if add_lots <= 0:
                add_lots = min(
                    cfg.default_lot_size * 0.5,
                    cfg.max_lot_size - position.lots,
                )

            if add_lots >= 0.01:
                lifecycle = self._get_lifecycle_state(position)
                reasons.append(
                    f"📈 SCALE UP [{lifecycle.value.upper()}]: +€{position.unrealized_pnl:.2f} profit "
                    f"({r_mult:.1f}R), strong signal ({signal_strength:.2f})"
                )
                return self._make_decision(
                    action=PositionAction.SCALE_UP,
                    symbol=symbol,
                    lots=round(add_lots, 2),
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                )

        # Smart scale-down of losers is handled separately and is optional.
        scale_result = self._smart_scale_down_common(
            position=position,
            expert_signal_strength=signal_strength,
            support_ratio=None,
        )

        if scale_result is not None:
            action, reduce_lots, scale_reason = scale_result
            reasons.append(scale_reason)
            return self._make_decision(
                action=action,
                symbol=symbol,
                lots=reduce_lots,
                side=position.side,
                confidence=0.65,
                reasons=reasons,
            )

        lifecycle = self._get_lifecycle_state(position)
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
        reasons.append(
            f"HOLD [{lifecycle.value.upper()}] (aligning): {position.direction} {position.lots:.2f} lots, "
            f"P&L €{position.unrealized_pnl:.2f} ({r_mult:.1f}R)"
        )
        return self._make_decision(
            action=PositionAction.HOLD,
            symbol=symbol,
            confidence=0.60,
            reasons=reasons,
        )

    # =========================================================
    # SMART SCALE DOWN (OPTIONAL, LOSING POSITIONS)
    # =========================================================

    def _smart_scale_down_common(
        self,
        position: LivePosition,
        expert_signal_strength: float,
        support_ratio: Optional[float] = None,
    ) -> Optional[Tuple[PositionAction, float, str]]:
        """
        Unified smart scale-down logic used by both _decide_scale() and manage_position().

        IMPORTANT:
        - This is **only** for LOSING positions.
        - It is gated by config.enable_smart_scale_down (default: False).
        - Full exits remain the responsibility of ExitStrategyEngine.
        """
        # Feature gate: by default we DO NOT scale down losers at all.
        if not getattr(self.config, "enable_smart_scale_down", False):
            return None

        symbol = position.symbol
        cfg = self._get_symbol_cfg(symbol)

        if position.lots <= 0.01:
            return None

        if position.unrealized_pnl >= 0:
            # Only manage losers here; winners handled elsewhere.
            return None

        lifecycle = self._get_lifecycle_state(position)
        aggression = self._get_aggression_multiplier()

        use_r_based = self.config.use_r_based_scaling

        if use_r_based:
            r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
            loss_metric = r_mult  # negative
            trigger_threshold = cfg.get(
                "scale_down_trigger_r", self.config.scale_down_trigger_r
            )
            aggressive_threshold = cfg.get(
                "scale_down_aggressive_r", self.config.scale_down_aggressive_r
            )
            emergency_threshold = cfg.get(
                "scale_down_emergency_r", self.config.scale_down_emergency_r
            )
            metric_name = "R"
        else:
            loss_amount = min(0.0, position.unrealized_pnl)  # negative
            trigger_loss = cfg.get(
                "scale_down_trigger_loss_eur", self.config.scale_down_trigger_loss_eur
            )
            loss_metric = loss_amount
            trigger_threshold = -trigger_loss
            aggressive_threshold = -trigger_loss * 1.5
            emergency_threshold = -trigger_loss * 2.0
            metric_name = "EUR"

        conviction = support_ratio if support_ratio is not None else expert_signal_strength
        expert_weakening = conviction < 0.40
        expert_uncertain = conviction < 0.55

        should_scale_down = False
        scale_reason = ""
        reduce_pct = 0.0

        # Lifecycle adjustments – more strict in PROBE/DEFEND, more patient in RIDE.
        if lifecycle == PositionLifecycle.PROBE:
            trigger_threshold *= 0.7
        elif lifecycle == PositionLifecycle.RIDE:
            trigger_threshold *= 1.3
        elif lifecycle == PositionLifecycle.DEFEND:
            trigger_threshold *= 0.8

        # TIER 1: Not yet at trigger
        if loss_metric > trigger_threshold:
            should_scale_down = False

        # TIER 2: Between trigger and aggressive – only if experts weakening
        elif loss_metric > aggressive_threshold:
            if expert_weakening:
                should_scale_down = True
                reduce_pct = 0.3 * aggression
                scale_reason = f"experts weakening ({conviction:.2f})"
            elif expert_uncertain and loss_metric <= (trigger_threshold + aggressive_threshold) / 2.0:
                should_scale_down = True
                reduce_pct = 0.4 * aggression
                scale_reason = f"experts uncertain ({conviction:.2f}) + growing loss"

        # TIER 3: Between aggressive and emergency – stronger reduction
        elif loss_metric > emergency_threshold:
            should_scale_down = True
            reduce_pct = 0.5 * aggression
            if expert_weakening:
                scale_reason = f"aggressive zone + weak experts ({conviction:.2f})"
            else:
                scale_reason = f"aggressive zone, loss={loss_metric:.2f}{metric_name}"

        # TIER 4: Beyond emergency threshold – heavy scale-down (not full exit)
        else:
            should_scale_down = True
            reduce_pct = 0.6  # keep some size; ExitEngine still in charge of hard stop
            scale_reason = (
                f"EMERGENCY: loss={loss_metric:.2f}{metric_name} <= emergency threshold"
            )

        if should_scale_down and reduce_pct > 0.0:
            reduce_lots = min(position.lots * reduce_pct, position.lots - 0.01)
            if reduce_lots >= 0.01:
                lifecycle_tag = (
                    f"[{lifecycle.value.upper()}]" if lifecycle != PositionLifecycle.BUILD else ""
                )
                full_reason = (
                    f"🛡️ SMART SCALE DOWN {lifecycle_tag}: "
                    f"{loss_metric:.2f}{metric_name} loss, {scale_reason}, "
                    f"reducing {reduce_pct*100:.0f}%"
                )
                return (PositionAction.SCALE_DOWN, round(reduce_lots, 2), full_reason)

        return None

    # =========================================================
    # COOPERATIVE MANAGEMENT MODE (manage_position)
    # =========================================================

    def manage_position(
        self,
        signal: PositionManagementSignal,
    ) -> SmartDecision:
        """
        Position-focused decision making with full expert context.

        - Exit rules always flow through ExitStrategyEngine.
        - Experts modulate scaling and SL tightening in non-critical situations.
        """
        symbol = signal.symbol
        position = self._positions.get(symbol)
        cfg = self.config
        sym_cfg = self._get_symbol_cfg(symbol)
        reasons: List[str] = []

        # No position = fallback to simple decide()
        if position is None:
            direction = (
                1
                if signal.consensus_action == "BUY"
                else (-1 if signal.consensus_action == "SELL" else 0)
            )
            return self.decide(
                symbol=symbol,
                signal_direction=direction,
                signal_strength=signal.consensus_confidence,
                consensus_confidence=signal.consensus_score,
            )

        # Update local peak
        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        lifecycle = self._get_lifecycle_state(position)
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)

        # Expert alignment with current position
        position_action = "BUY" if position.side > 0 else "SELL"
        supporting_experts: List[ExpertSignal] = []
        opposing_experts: List[ExpertSignal] = []

        for expert in signal.expert_signals:
            if expert.action == position_action or expert.action == "HOLD":
                supporting_experts.append(expert)
            else:
                opposing_experts.append(expert)

        total_experts = len(signal.expert_signals)
        support_ratio = (
            len(supporting_experts) / total_experts if total_experts > 0 else 0.5
        )

        management_ctx: Dict[str, Any] = {
            "support_ratio": support_ratio,
            "supporting_experts": [e.expert_name for e in supporting_experts],
            "opposing_experts": [e.expert_name for e in opposing_experts],
            "consensus_action": signal.consensus_action,
            "consensus_confidence": signal.consensus_confidence,
            "fragility": signal.fragility,
            "regime": signal.regime,
            "position_pnl": position.unrealized_pnl,
            "position_age_hours": position.age_hours,
        }

        # ═══════════════════════════════════════════════════════════════════
        # FINAL EXIT WINDOW CHECK (manage_position path)
        # Same logic as in decide() - exit positions before market close
        # unless loss is too large.
        # ═══════════════════════════════════════════════════════════════════
        in_final_exit, is_obligatory, max_loss_pct = self._check_final_exit_window()
        if in_final_exit and is_obligatory:
            balance = self._get_current_balance()
            max_loss_amount = balance * max_loss_pct
            
            if position.unrealized_pnl >= 0:
                # Position is profitable or breakeven - exit to avoid overnight
                reasons.append(
                    f"🌙 FINAL EXIT WINDOW: Closing profitable/breakeven position "
                    f"(P&L €{position.unrealized_pnl:.2f}) to avoid overnight exposure"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.80,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )
            elif abs(position.unrealized_pnl) <= max_loss_amount:
                # Position has acceptable loss - exit to avoid overnight
                reasons.append(
                    f"🌙 FINAL EXIT WINDOW: Closing position with acceptable loss "
                    f"(P&L €{position.unrealized_pnl:.2f} within €{max_loss_amount:.2f} limit)"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )
            else:
                # Loss is too large - add warning but continue with normal evaluation
                reasons.append(
                    f"⚠️ FINAL EXIT WINDOW: Large loss €{position.unrealized_pnl:.2f} "
                    f"exceeds limit - continuing normal evaluation"
                )

        # ═══════════════════════════════════════════════════════════════════
        # PRIORITY -1: BREAKEVEN SL CHECK (runs BEFORE ExitEngine)
        # This ensures SL is moved to breakeven BEFORE trailing exit triggers.
        # Without this, profit could retrace and ExitEngine would close before
        # we had a chance to protect with breakeven SL.
        # ═══════════════════════════════════════════════════════════════════
        breakeven_threshold = cfg.breakeven_activation_eur
        if position.unrealized_pnl >= breakeven_threshold:
            new_sl = self._calculate_breakeven_sl(position)
            if new_sl is not None and self._is_sl_improvement(position, new_sl):
                reasons.append(
                    f"🔒 BREAKEVEN FIRST: Profit €{position.unrealized_pnl:.2f} >= €{breakeven_threshold:.0f} "
                    f"→ moving SL to breakeven before exit evaluation"
                )
                return self._make_decision(
                    action=PositionAction.ADJUST_SL,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                    new_sl=new_sl,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

        # PRIORITY 0: unified ExitEngine
        if signal.consensus_action == "BUY":
            signal_dir = 1
        elif signal.consensus_action == "SELL":
            signal_dir = -1
        else:
            signal_dir = 0

        exit_ctx = PositionContext(
            symbol=symbol,
            side=position.side,
            unrealized_pnl=position.unrealized_pnl,
            peak_pnl=self._profit_peaks.get(symbol, position.unrealized_pnl),
            entry_price=position.entry_price,
            current_price=position.current_price,
            open_time=position.open_time,
            lots=position.lots,
            position_id=str(position.ticket) if hasattr(position, "ticket") else "",
            regime=signal.regime or "auto",
            signal_direction=signal_dir,
            signal_strength=float(max(0.0, min(1.0, signal.consensus_confidence))),
            signal_valid=self._is_signal_valid_for_symbol(symbol),
            consensus_confidence=float(max(0.0, min(1.0, signal.consensus_score))),
        )

        exit_engine = get_exit_engine()
        exit_decision = exit_engine.evaluate(exit_ctx)

        management_ctx["exit_engine_reason"] = exit_decision.reason.name
        management_ctx["exit_engine_should_exit"] = exit_decision.should_exit
        management_ctx["exit_engine_confidence"] = exit_decision.confidence
        management_ctx["lifecycle_state"] = lifecycle.value
        management_ctx["r_multiple"] = r_mult

        if exit_decision.should_exit:
            engine_msg = exit_decision.details.get("message", exit_decision.reason.name)
            reasons.append(f"[ExitEngine] {engine_msg}")

            if exit_decision.is_critical:
                reasons.append(
                    "Critical exit signalled by ExitEngine; overriding expert opinions."
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=exit_decision.confidence,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            # ═══════════════════════════════════════════════════════════════════
            # PPO MASTER EXIT / REVERSAL CHECK (v5.2+)
            # If PPO explicitly wants to close/reverse, respect it over experts
            # ═══════════════════════════════════════════════════════════════════
            side_str = "long" if position.side > 0 else "short"
            ppo_view = self._get_ppo_decision(symbol, side_str)

            ppo_exit_threshold = getattr(
                self.config,
                "ppo_exit_conf_threshold",
                getattr(self.config, "ppo_reversal_conf_threshold", 0.6),
            )

            # 1) Hard PPO explicit close
            if ppo_view.explicit_close and ppo_view.confidence >= ppo_exit_threshold:
                reasons.append(
                    f"🎯 PPO explicit CLOSE intent (conf={ppo_view.confidence:.2f})"
                )
                self.logger.info(
                    f"[PPO_EXIT] {symbol}: PPO explicit CLOSE - "
                    f"overriding expert support {support_ratio * 100:.0f}%"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=max(exit_decision.confidence, ppo_view.confidence),
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            # 2) PPO reversal
            if ppo_view.is_reversal and ppo_view.confidence >= ppo_exit_threshold:
                should_respect, ppo_reason = self._should_respect_ppo_reversal(
                    symbol, ppo_view.confidence, position.unrealized_pnl
                )
                if should_respect:
                    reasons.append(f"🎯 {ppo_reason}")
                    self.logger.info(
                        f"[PPO_EXIT] {symbol}: {ppo_reason} - "
                        f"closing position (overriding expert support {support_ratio * 100:.0f}%)"
                    )
                    return self._make_decision(
                        action=PositionAction.CLOSE,
                        symbol=symbol,
                        side=position.side,
                        confidence=max(exit_decision.confidence, ppo_view.confidence),
                        reasons=reasons,
                        management_context=management_ctx,
                        expert_support_ratio=support_ratio,
                    )
                else:
                    reasons.append(f"⏸️ PPO reversal signal too weak: {ppo_reason}")

            # Non-critical: experts may influence whether we tighten or close
            if support_ratio <= 0.3 or position.unrealized_pnl <= 0:
                reasons.append(
                    f"Non-critical exit aligned with experts "
                    f"({int((1.0 - support_ratio) * 100)}% against or flat/losing)."
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=exit_decision.confidence,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            new_sl = self._calculate_tightened_sl(position, signal)
            if new_sl is not None and self._is_sl_improvement(position, new_sl):
                reasons.append(
                    "ExitEngine suggests non-critical exit, but experts support "
                    "the position and it is profitable – tightening SL instead."
                )
                return self._make_decision(
                    action=PositionAction.TIGHTEN_PROTECTION,
                    symbol=symbol,
                    side=position.side,
                    confidence=max(exit_decision.confidence, 0.7),
                    reasons=reasons,
                    new_sl=new_sl,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            reasons.append(
                "Unable to compute a better protective SL – "
                "respecting ExitEngine non-critical exit."
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=exit_decision.confidence,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=support_ratio,
            )

        # ExitEngine wants HOLD – cooperative management mode

        # ═══════════════════════════════════════════════════════════════════
        # PPO MASTER EXIT / REVERSAL CHECK (v5.2+) - Even when ExitEngine says HOLD
        # If PPO explicitly wants to close/reverse this instrument, respect it.
        # This ensures PPO has a voice in exit decisions.
        # ═══════════════════════════════════════════════════════════════════
        side_str = "long" if position.side > 0 else "short"
        ppo_view = self._get_ppo_decision(symbol, side_str)

        ppo_exit_threshold = getattr(
            self.config,
            "ppo_exit_conf_threshold",
            getattr(self.config, "ppo_reversal_conf_threshold", 0.6),
        )

        # 1) Hard PPO explicit close: direction can be FLAT, but PPO
        #    explicitly wants this instrument closed.
        if ppo_view.explicit_close and ppo_view.confidence >= ppo_exit_threshold:
            reasons.append(
                f"🎯 PPO explicit CLOSE intent (conf={ppo_view.confidence:.2f})"
            )
            reasons.append(
                f"PPO MASTER: Overriding ExitEngine HOLD and expert support "
                f"({support_ratio:.0%}) for explicit close"
            )
            self.logger.info(
                f"[PPO_EXIT] {symbol}: PPO explicit CLOSE (conf={ppo_view.confidence:.3f}) - "
                f"overriding ExitEngine HOLD, experts={support_ratio * 100:.0f}%"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=ppo_view.confidence,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=support_ratio,
            )

        # 2) PPO reversal: wants to flip side (close + re-enter opposite).
        #    PositionManager only handles the CLOSE; entry logic will
        #    see PPO's new direction on the next cycle.
        if ppo_view.is_reversal and ppo_view.confidence >= ppo_exit_threshold:
            should_respect, ppo_reason = self._should_respect_ppo_reversal(
                symbol, ppo_view.confidence, position.unrealized_pnl
            )
            if should_respect:
                reasons.append(f"🎯 {ppo_reason}")
                reasons.append(
                    f"PPO MASTER: Overriding ExitEngine HOLD and expert support "
                    f"({support_ratio:.0%}) for reversal {side_str} → {ppo_view.direction}"
                )
                self.logger.info(
                    f"[PPO_EXIT] {symbol}: PPO reversal {side_str} → {ppo_view.direction} "
                    f"(conf={ppo_view.confidence:.3f}) - overriding ExitEngine HOLD, "
                    f"experts={support_ratio * 100:.0f}%"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=ppo_view.confidence,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )
            else:
                # Log weak PPO reversal signal for debugging
                self.logger.debug(
                    f"[PPO_EXIT] {symbol}: Weak reversal signal - {ppo_reason}"
                )

        # Get per-instrument thresholds (respects XAUUSD/EURUSD overrides from risk_policy.yaml)
        sym_cfg = self._get_symbol_cfg(symbol)
        scale_up_min_profit = sym_cfg.get("scale_up_min_profit_eur", cfg.scale_up_min_profit_eur)
        scale_up_min_r = sym_cfg.get("scale_up_min_r", cfg.scale_up_min_r)
        use_r_based = cfg.use_r_based_scaling
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)

        # Determine if profit is sufficient for scaling (R-based or EUR-based)
        profit_ok_for_scale = (
            (use_r_based and r_mult >= scale_up_min_r)
            or (not use_r_based and position.unrealized_pnl >= scale_up_min_profit)
        )

        # PRIORITY 1: strong expert support + profit
        if support_ratio >= 0.6 and position.unrealized_pnl > 0:
            # 1a) scale up with strong support + good profit
            # NOTE: Scale-up is DISABLED by default (enable_scale_up=false)
            # because MT5 creates new tickets instead of modifying existing positions.
            if (
                cfg.enable_scale_up  # Must be explicitly enabled
                and support_ratio >= 0.75
                and profit_ok_for_scale
            ):
                last_scale = self._last_scale_time.get(symbol, 0.0)
                if (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds:
                    if position.lots < cfg.max_lot_size:
                        add_lots = self._calculate_scale_lots(position, signal)
                        if add_lots >= 0.01:
                            reasons.append(
                                f"📈 SCALE UP: {int(support_ratio * 100)}% expert support, "
                                f"+€{position.unrealized_pnl:.2f} profit ({r_mult:.1f}R), "
                                f"threshold={scale_up_min_r:.1f}R/€{scale_up_min_profit:.0f}"
                            )
                            return self._make_decision(
                                action=PositionAction.SCALE_UP,
                                symbol=symbol,
                                lots=add_lots,
                                side=position.side,
                                confidence=0.70,
                                reasons=reasons,
                                management_context=management_ctx,
                                expert_support_ratio=support_ratio,
                            )

            # 1b) tighten SL further beyond breakeven when profit is high
            # Note: Breakeven already set at PRIORITY -1; this tightens further
            # Only if profit significantly above breakeven threshold
            if position.unrealized_pnl >= breakeven_threshold * 1.5:  # €225+ for €150 threshold
                new_sl = self._calculate_tightened_sl(position, signal)
                if new_sl is not None and self._is_sl_improvement(position, new_sl):
                    reasons.append(
                        f"🔒 TIGHTEN FURTHER: High profit €{position.unrealized_pnl:.2f} "
                        f"({int(support_ratio * 100)}% expert support) → locking more gains"
                    )
                    return self._make_decision(
                        action=PositionAction.TIGHTEN_PROTECTION,
                        symbol=symbol,
                        side=position.side,
                        confidence=0.65,
                        reasons=reasons,
                        new_sl=new_sl,
                        management_context=management_ctx,
                        expert_support_ratio=support_ratio,
                    )

        # PRIORITY 2: optional smart scale-down of losers (if enabled)
        scale_result = self._smart_scale_down_common(
            position=position,
            expert_signal_strength=signal.consensus_confidence,
            support_ratio=support_ratio,
        )

        if scale_result is not None:
            action, reduce_lots, scale_reason = scale_result
            reasons.append(scale_reason)
            return self._make_decision(
                action=action,
                symbol=symbol,
                lots=reduce_lots,
                side=position.side,
                confidence=0.65,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=support_ratio,
            )

        # DEFAULT: HOLD
        reasons.append(
            f"✓ HOLD [{lifecycle.value.upper()}]: {position.direction} {position.lots:.2f} lots | "
            f"PnL €{position.unrealized_pnl:.2f} ({r_mult:.1f}R) | "
            f"{int(support_ratio * 100)}% expert support | "
            f"age {position.age_hours:.1f}h"
        )
        return self._make_decision(
            action=PositionAction.HOLD,
            symbol=symbol,
            confidence=0.60,
            reasons=reasons,
            management_context=management_ctx,
            expert_support_ratio=support_ratio,
        )

    # =========================================================
    # SL / SCALE HELPERS
    # =========================================================

    def _calculate_tightened_sl(
        self,
        position: LivePosition,
        signal: PositionManagementSignal,
    ) -> Optional[float]:
        """Calculate a tightened stop loss to protect profit when experts turn against."""
        try:
            if position.current_price <= 0 or position.entry_price <= 0:
                return None

            profit_per_unit = position.current_price - position.entry_price
            if position.side < 0:  # SHORT
                profit_per_unit = position.entry_price - position.current_price

            protection_buffer = abs(profit_per_unit) * 0.5

            if position.side > 0:  # LONG
                new_sl = position.current_price - protection_buffer
            else:  # SHORT
                new_sl = position.current_price + protection_buffer

            return round(new_sl, 5)
        except Exception:
            return None

    def _calculate_breakeven_sl(self, position: LivePosition) -> Optional[float]:
        """Calculate SL at breakeven (entry price + small buffer)."""
        try:
            if position.entry_price <= 0:
                return None

            buffer = position.entry_price * 0.0002  # ~2 pips buffer

            if position.side > 0:  # LONG
                return round(position.entry_price + buffer, 5)
            else:  # SHORT
                return round(position.entry_price - buffer, 5)
        except Exception:
            return None

    def _is_sl_improvement(self, position: LivePosition, new_sl: float) -> bool:
        """Check if new SL is better (more protective) than current."""
        try:
            if position.sl <= 0:
                return True

            if position.side > 0:  # LONG: higher SL is better
                return new_sl > position.sl
            else:  # SHORT: lower SL is better
                return new_sl < position.sl
        except Exception:
            return False

    def _calculate_scale_lots(
        self,
        position: LivePosition,
        signal: PositionManagementSignal,
    ) -> float:
        """Calculate lots to add when scaling up based on expert support."""
        try:
            cfg = self.config

            if self.lot_calculator and cfg.use_unified_calculator:
                scale_signal = signal.consensus_confidence * signal.support_ratio * 0.5
                add_lots, _ = self.lot_calculator.calculate_lots(
                    symbol=position.symbol,
                    signal_strength=scale_signal,
                )
            else:
                add_lots = cfg.default_lot_size * signal.support_ratio * 0.5

            max_add = cfg.max_lot_size - position.lots
            add_lots = min(add_lots, max_add, cfg.default_lot_size)

            return round(max(0.01, add_lots), 2)
        except Exception:
            return 0.0

    # =========================================================
    # PUBLIC UTILS
    # =========================================================

    def record_trade(self, symbol: str, is_scale: bool = False) -> None:
        """Record that a trade was made for cooldown tracking."""
        now = time.time()
        self._last_trade_time[symbol] = now
        if is_scale:
            self._last_scale_time[symbol] = now

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        """Get current position for a symbol."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, LivePosition]:
        """Get all current positions."""
        return self._positions.copy()

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position on a symbol."""
        return symbol in self._positions

    def get_net_exposure(self) -> Dict[str, float]:
        """Get net exposure per symbol in lots."""
        return {symbol: pos.lots * pos.side for symbol, pos in self._positions.items()}

    def get_total_pnl(self) -> float:
        """Get total unrealized P&L across all positions."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def needs_hedge_cleanup(self, mt5_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify positions that should be closed to eliminate hedging.

        Returns list of position dicts that should be closed.
        """
        to_close: List[Dict[str, Any]] = []
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}

        for pos in mt5_positions:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol.setdefault(symbol, []).append(pos)

        for symbol, positions in by_symbol.items():
            if len(positions) <= 1:
                continue

            buys = [p for p in positions if p.get("type", 0) == 0]
            sells = [p for p in positions if p.get("type", 0) == 1]

            if buys and sells:
                buy_lots = sum(float(p.get("volume", 0.0) or 0.0) for p in buys)
                sell_lots = sum(float(p.get("volume", 0.0) or 0.0) for p in sells)

                if buy_lots < sell_lots:
                    to_close.extend(buys)
                    self.logger.warning(
                        format_operator_message(
                            "🔄",
                            "HEDGE_CLEANUP",
                            symbol=symbol,
                            action="Closing BUY side",
                            buy_lots=f"{buy_lots:.2f}",
                            sell_lots=f"{sell_lots:.2f}",
                        )
                    )
                else:
                    to_close.extend(sells)
                    self.logger.warning(
                        format_operator_message(
                            "🔄",
                            "HEDGE_CLEANUP",
                            symbol=symbol,
                            action="Closing SELL side",
                            buy_lots=f"{buy_lots:.2f}",
                            sell_lots=f"{sell_lots:.2f}",
                        )
                    )

        return to_close

    def needs_consolidation(self, mt5_positions: List[Dict[str, Any]]) -> bool:
        """Check if positions need consolidation (multiple tickets same direction)."""
        by_symbol: Dict[str, int] = {}
        for pos in mt5_positions:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

        return any(
            count > self.config.max_positions_per_symbol
            for count in by_symbol.values()
        )

    def get_actual_position_count(self) -> int:
        """
        Get the actual MT5 position count (before netting).
        """
        return self._actual_mt5_position_count

    def has_hedged_positions(self) -> bool:
        """
        Check if we have hedged positions (actual count > net count).
        """
        return self._actual_mt5_position_count > len(self._positions)

    def log_status(self) -> None:
        """Log current position status."""
        if not self._positions:
            self.logger.info("📊 No open positions")
            return

        total_pnl = self.get_total_pnl()

        self.logger.info(
            format_operator_message(
                "📊",
                "POSITION_STATUS",
                count=len(self._positions),
                total_pnl=f"€{total_pnl:.2f}",
            )
        )

        for symbol, pos in self._positions.items():
            self.logger.info(
                f"  {symbol}: {pos.direction} {pos.lots:.2f} lots @ {pos.entry_price:.5f} "
                f"| P&L: €{pos.unrealized_pnl:.2f} | Age: {pos.age_hours:.1f}h"
            )
