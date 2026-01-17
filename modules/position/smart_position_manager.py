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
#
# Enhancements in this version:
#   - Single cached YAML loader (no repeated disk reads, consistent overrides)
#   - Correct SL/TP selection when netting tickets (choose representative ticket)
#   - Breakeven + profit-lock ladder available in BOTH decide() and manage_position()
#   - Reduced duplication via shared helpers (final-exit, PPO overrides, SL logic)
#   - Safer parsing + defensive guards (never block trading on non-critical failures)
# -------------------------------------------------------------

from __future__ import annotations

import time
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
from functools import lru_cache

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
# YAML CONFIG LOADING (CACHED + CONSISTENT)
# =========================================================

_RISK_POLICY_PATH = Path(__file__).resolve().parents[2] / "config" / "risk_policy.yaml"


@lru_cache(maxsize=1)
def _load_risk_policy_yaml_cached(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _load_risk_policy_yaml() -> Dict[str, Any]:
    try:
        return _load_risk_policy_yaml_cached(str(_RISK_POLICY_PATH))
    except Exception:
        return {}


def refresh_risk_policy_cache() -> None:
    """Clear cached risk_policy.yaml. Useful if you hot-edit YAML during runtime."""
    try:
        _load_risk_policy_yaml_cached.cache_clear()
    except Exception:
        pass


def load_config_from_yaml() -> Dict[str, Any]:
    """Load smart position config block from risk_policy.yaml."""
    try:
        policy = _load_risk_policy_yaml()
        smart_pos = policy.get("smart_position", {}) or {}
        return smart_pos if isinstance(smart_pos, dict) else {}
    except Exception:
        return {}


def load_lot_config_from_yaml() -> Dict[str, Any]:
    """Load unified lot sizing config from risk_policy.yaml."""
    try:
        policy = _load_risk_policy_yaml()
        lot = policy.get("lot_sizing", {}) or {}
        return lot if isinstance(lot, dict) else {}
    except Exception:
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
        return sum(1 for e in self.expert_signals if e.supports_position)

    @property
    def experts_against_position(self) -> int:
        return sum(1 for e in self.expert_signals if not e.supports_position)

    @property
    def support_ratio(self) -> float:
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

    Published to SmartInfoBus as "position_focus_context" by Executor.
    """
    focus_mode_active: bool = False
    positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)

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

    total_exposure: float = 0.0
    risk_level: str = "normal"  # normal, elevated, critical
    evaluation_mode: str = "position_management"  # "new_signals" | "position_management"

    def has_position(self, instrument: str) -> bool:
        return instrument in self.positions

    def get_position_side(self, instrument: str) -> int:
        pos = self.positions.get(instrument)
        if not pos:
            return 0
        return int(pos.get("side", 0))

    def should_experts_focus_on_position(self) -> bool:
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
        if not positions:
            return cls(focus_mode_active=False)

        # Find primary position by notional
        primary_inst = None
        primary_notional = 0.0
        for inst, pos in positions.items():
            notional = abs(float(pos.get("notional_eur", 0.0) or (pos.get("units", 0.0) * pos.get("entry_price", 1.0))))
            if notional > primary_notional:
                primary_notional = notional
                primary_inst = inst

        ctx = cls(
            focus_mode_active=True,
            positions=positions,
            primary_instrument=primary_inst,
            evaluation_mode="position_management",
        )

        # Fill primary position fields
        if primary_inst and primary_inst in positions:
            pos = positions[primary_inst]
            ctx.primary_side = int(pos.get("side", 0))
            ctx.primary_entry_price = float(pos.get("entry_price", 0.0))
            ctx.primary_current_price = float(pos.get("current_price", pos.get("entry_price", 0.0)))
            ctx.primary_pnl = float(pos.get("unrealized_pnl", 0.0))
            ctx.primary_lots = float(pos.get("units", pos.get("lots", 0.0)))
            ctx.primary_ticket = int(pos.get("ticket", 0))
            ctx.primary_sl = float(pos.get("sl", 0.0))
            ctx.primary_tp = float(pos.get("tp", 0.0))

            if ctx.primary_entry_price > 0 and ctx.primary_lots > 0:
                price_diff = ctx.primary_current_price - ctx.primary_entry_price
                if ctx.primary_side < 0:
                    price_diff = -price_diff
                ctx.primary_pnl_pct = (price_diff / ctx.primary_entry_price) * 100

            open_time = pos.get("open_time", 0)
            if open_time:
                from datetime import datetime
                if isinstance(open_time, str):
                    try:
                        dt = datetime.fromisoformat(open_time.replace("Z", "+00:00"))
                        open_ts = dt.timestamp()
                    except Exception:
                        open_ts = 0.0
                else:
                    try:
                        open_ts = float(open_time)
                    except Exception:
                        open_ts = 0.0

                if open_ts > 0:
                    ctx.primary_age_hours = (time.time() - open_ts) / 3600.0

        ctx.total_exposure = sum(
            abs(float(p.get("notional_eur", 0.0) or (float(p.get("units", 0.0)) * float(p.get("entry_price", 1.0)))))
            for p in positions.values()
        )
        return ctx


@dataclass
class LivePosition:
    """Represents a live net position from MT5."""
    symbol: str
    side: int  # 1 = BUY, -1 = SELL
    lots: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    open_time: float  # Unix timestamp (seconds)
    ticket: int = 0
    sl: float = 0.0
    tp: float = 0.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.open_time if self.open_time > 0 else 0.0

    @property
    def age_hours(self) -> float:
        return self.age_seconds / 3600.0

    @property
    def is_profitable(self) -> bool:
        return self.unrealized_pnl > 0.0

    @property
    def direction(self) -> str:
        return "BUY" if self.side > 0 else "SELL"


@dataclass
class PPODecisionView:
    """
    Normalised view of Arbiter → PPO decision for one symbol.

    Version: v5.2 - Supports PPO explicit close/reverse intents from ArbiterLogic
    """
    direction: str = "flat"          # "long" / "short" / "flat"
    confidence: float = 0.0          # 0–1
    position_size: float = 0.0       # 0–1
    action_intent: str = ""          # "open_long", "close", "reverse", ...

    # Discrete-action metadata (MaskablePPO): HOLD vs CLOSE are distinct in training,
    # but both map to a (0,0) continuous proxy. Use this to respect true CLOSE only.
    discrete_intent: str = ""        # "hold" | "long" | "short" | "close"
    discrete_action_id: int = -1

    explicit_close: bool = False
    explicit_reverse: bool = False

    wants_flat: bool = False
    is_reversal: bool = False

    @property
    def direction_int(self) -> int:
        if self.direction == "long":
            return 1
        if self.direction == "short":
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
    close_first: bool = False
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    expert_support_ratio: float = 0.5
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
    max_positions_per_symbol: int = 1
    max_total_positions: int = 4

    default_lot_size: float = 0.01
    max_lot_size: float = 50.0
    use_unified_calculator: bool = True

    scale_up_min_profit_eur: float = 50.0
    scale_up_cooldown_seconds: float = 600.0
    scale_down_trigger_loss_eur: float = 40.0

    enable_scale_up: bool = True

    breakeven_activation_eur: float = 150.0

    same_direction_cooldown_seconds: float = 300.0
    reversal_cooldown_seconds: float = 300.0

    min_signal_strength: float = 0.10
    strong_signal_threshold: float = 0.50
    reversal_signal_threshold: float = 0.45

    ppo_exit_conf_threshold: float = 0.60
    ppo_reversal_conf_threshold: float = 0.60  # legacy alias

    enable_smart_scale_down: bool = False

    # R-based scaling
    use_r_based_scaling: bool = True
    scale_up_min_r: float = 1.0
    scale_down_trigger_r: float = -0.5
    scale_down_aggressive_r: float = -1.0
    scale_down_emergency_r: float = -1.5

    # Lifecycle
    probe_max_age_hours: float = 0.5
    probe_max_profit_r: float = 0.3
    build_max_age_hours: float = 2.0
    build_max_profit_r: float = 1.0
    ride_min_profit_r: float = 0.5
    defend_trigger_drawdown_pct: float = 0.20

    # Drawdown-aware aggression
    dd_aggression_enabled: bool = True
    mild_dd_threshold: float = 0.015
    moderate_dd_threshold: float = 0.025
    severe_dd_threshold: float = 0.035
    mild_aggression_mult: float = 0.85
    moderate_aggression_mult: float = 0.65
    severe_aggression_mult: float = 0.40

    # Startup grace
    default_startup_grace_calls: int = 3

    # Adaptive protection (beyond breakeven)
    enable_profit_lock: bool = True
    profit_lock_min_r: float = 1.2
    profit_lock_min_eur_mult: float = 1.5  # relative to breakeven_activation_eur
    # (r_threshold, lock_fraction_of_open_profit)
    profit_lock_ladder: Tuple[Tuple[float, float], ...] = (
        (1.2, 0.10),
        (1.6, 0.18),
        (2.0, 0.28),
        (2.6, 0.40),
        (3.5, 0.55),
    )


# ═══════════════════════════════════════════════════════════════════════
# POSITION LIFECYCLE STATE MACHINE (v4.0)
# ═══════════════════════════════════════════════════════════════════════

class PositionLifecycle(Enum):
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
      - Net MT5 tickets into 1 logical position per symbol
      - Delegate exits to ExitStrategyEngine (single source of truth)
      - Respect PPO explicit exit/reversal intents (PPO is master)
      - Optional: breakeven + adaptive profit-lock stop management
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = SmartPositionConfig()

        self.logger = RotatingLogger(
            "SmartPositionManager",
            log_path="logs/position/smart_manager.log",
            operator_mode=True,
            max_lines=5000,
        )

        # InfoBus for state persistence
        try:
            from modules.utils.info_bus import InfoBusManager
            self._smart_bus = InfoBusManager.get_instance()
        except Exception:
            self._smart_bus = None

        # Load YAML if no external config provided
        if config is None:
            config = load_config_from_yaml()

        # Apply scalar top-level overrides
        if isinstance(config, dict) and config:
            for key, value in config.items():
                if isinstance(value, dict):
                    continue
                if hasattr(self.config, key):
                    try:
                        setattr(self.config, key, value)
                    except Exception:
                        pass

            # Optional advanced adaptive lock ladder from YAML:
            # smart_position:
            #   profit_lock:
            #     enabled: true
            #     min_r: 1.2
            #     ladder: [[1.2, 0.10], [1.6, 0.18], ...]
            profit_lock = config.get("profit_lock", {})
            if isinstance(profit_lock, dict):
                if profit_lock.get("enabled") is not None:
                    self.config.enable_profit_lock = bool(profit_lock.get("enabled"))
                raw_min_r = profit_lock.get("min_r")
                if isinstance(raw_min_r, (int, float, str)):
                    try:
                        self.config.profit_lock_min_r = float(raw_min_r)
                    except Exception:
                        pass
                ladder = profit_lock.get("ladder")
                if isinstance(ladder, list):
                    parsed: List[Tuple[float, float]] = []
                    for row in ladder:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            try:
                                parsed.append((float(row[0]), float(row[1])))
                            except Exception:
                                continue
                    if parsed:
                        self.config.profit_lock_ladder = tuple(parsed)

        # Lot sizing overrides from unified lot sizing
        lot_config = load_lot_config_from_yaml()
        if isinstance(lot_config, dict) and lot_config:
            if "max_lot" in lot_config:
                try:
                    self.config.max_lot_size = float(lot_config["max_lot"])
                except Exception:
                    pass

        # Initialize unified lot calculator (lazy)
        self._lot_calculator = None

        # STATE TRACKING
        self._positions: Dict[str, LivePosition] = {}
        self._actual_mt5_position_count: int = 0
        self._profit_peaks: Dict[str, float] = self._load_peaks_from_bus()
        self._last_trade_time: Dict[str, float] = {}
        self._last_scale_time: Dict[str, float] = {}
        self._last_sync_time: float = 0.0

        self._decide_call_count_by_symbol: Dict[str, int] = {}
        self._decide_call_count: int = 0
        self._startup_grace_calls: int = 3  # deprecated

        self._initial_risk_by_symbol: Dict[str, float] = self._load_initial_risk_from_bus()
        self._lifecycle_states: Dict[str, PositionLifecycle] = {}

        # PPO decision noise suppression
        self._last_ppo_decision: Dict[str, str] = {}

        # Per-instrument overrides
        self._per_instrument_cfg: Dict[str, Dict[str, Any]] = self._load_per_instrument_config()

    # =========================================================
    # CONFIG HELPERS
    # =========================================================

    def _load_per_instrument_config(self) -> Dict[str, Dict[str, Any]]:
        """Load per-instrument configuration and lifecycle/DD overrides from YAML."""
        try:
            data = _load_risk_policy_yaml()
            smart_pos = data.get("smart_position", {}) or {}
            if not isinstance(smart_pos, dict):
                return {}

            per_inst = smart_pos.get("per_instrument", {}) or {}
            if not isinstance(per_inst, dict):
                per_inst = {}

            # Lifecycle overrides (flattened)
            lifecycle = smart_pos.get("lifecycle", {}) or {}
            if isinstance(lifecycle, dict):
                for key, val in lifecycle.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, val)

            # Drawdown aggression overrides
            dd_agg = smart_pos.get("drawdown_aggression", {}) or {}
            if isinstance(dd_agg, dict) and dd_agg:
                if dd_agg.get("enabled") is not None:
                    self.config.dd_aggression_enabled = bool(dd_agg["enabled"])
                for key in (
                    "mild_dd_threshold",
                    "moderate_dd_threshold",
                    "severe_dd_threshold",
                    "mild_aggression_mult",
                    "moderate_aggression_mult",
                    "severe_aggression_mult",
                ):
                    if key in dd_agg and hasattr(self.config, key):
                        try:
                            setattr(self.config, key, float(dd_agg[key]))
                        except Exception:
                            pass

            # R-based toggles
            for key in (
                "use_r_based_scaling",
                "scale_up_min_r",
                "scale_down_trigger_r",
                "scale_down_aggressive_r",
                "scale_down_emergency_r",
            ):
                if key in smart_pos and hasattr(self.config, key):
                    try:
                        val = smart_pos[key]
                        if isinstance(getattr(self.config, key), bool):
                            setattr(self.config, key, bool(val))
                        else:
                            setattr(self.config, key, float(val))
                    except Exception:
                        pass

            if "enable_smart_scale_down" in smart_pos:
                self.config.enable_smart_scale_down = bool(smart_pos.get("enable_smart_scale_down", False))

            if "enable_scale_up" in smart_pos:
                self.config.enable_scale_up = bool(smart_pos.get("enable_scale_up", False))

            # Profit lock tuning (optional)
            profit_lock = smart_pos.get("profit_lock", {})
            if isinstance(profit_lock, dict):
                if profit_lock.get("enabled") is not None:
                    self.config.enable_profit_lock = bool(profit_lock.get("enabled"))
                raw_min_r = profit_lock.get("min_r")
                if isinstance(raw_min_r, (int, float, str)):
                    try:
                        self.config.profit_lock_min_r = float(raw_min_r)
                    except Exception:
                        pass
                ladder = profit_lock.get("ladder")
                if isinstance(ladder, list):
                    parsed: List[Tuple[float, float]] = []
                    for row in ladder:
                        if isinstance(row, (list, tuple)) and len(row) >= 2:
                            try:
                                parsed.append((float(row[0]), float(row[1])))
                            except Exception:
                                continue
                    if parsed:
                        self.config.profit_lock_ladder = tuple(parsed)

            return per_inst

        except Exception as e:
            self.logger.warning(f"Failed to load per-instrument config: {e}")
            return {}

    def _get_symbol_cfg(self, symbol: str) -> Dict[str, Any]:
        """Single source of truth for per-symbol overrides merged over global defaults."""
        cfg: Dict[str, Any] = {
            "scale_up_min_profit_eur": self.config.scale_up_min_profit_eur,
            "scale_down_trigger_loss_eur": self.config.scale_down_trigger_loss_eur,
            "min_signal_strength": self.config.min_signal_strength,
            "strong_signal_threshold": self.config.strong_signal_threshold,
            "startup_grace_calls": self.config.default_startup_grace_calls,
            "scale_up_min_r": self.config.scale_up_min_r,
            "scale_down_trigger_r": self.config.scale_down_trigger_r,
            "scale_down_aggressive_r": self.config.scale_down_aggressive_r,
            "scale_down_emergency_r": self.config.scale_down_emergency_r,
            "probe_max_age_hours": self.config.probe_max_age_hours,
            "probe_max_profit_r": self.config.probe_max_profit_r,
            "build_max_age_hours": self.config.build_max_age_hours,
            "build_max_profit_r": self.config.build_max_profit_r,
            "ride_min_profit_r": self.config.ride_min_profit_r,
            "defend_trigger_drawdown_pct": self.config.defend_trigger_drawdown_pct,
            "breakeven_activation_eur": self.config.breakeven_activation_eur,
            "ppo_exit_conf_threshold": self.config.ppo_exit_conf_threshold,
        }

        per_inst = self._per_instrument_cfg.get(symbol)
        if isinstance(per_inst, dict):
            cfg.update(per_inst)

        return cfg

    # =========================================================
    # SEASONALITY / TRADING WINDOW CHECKS
    # =========================================================

    def _check_no_new_trades(self) -> Tuple[bool, str]:
        try:
            if self._smart_bus:
                seasonality = self._smart_bus.get("SeasonalityRiskExpert_voting_proposal", "SmartPositionManager")
                if isinstance(seasonality, dict):
                    trading_window = seasonality.get("trading_window", {})
                    if isinstance(trading_window, dict):
                        no_new = bool(trading_window.get("no_new_trades", False))
                        if no_new:
                            local_time = trading_window.get("local_time", "unknown")
                            minutes_to_close = trading_window.get("minutes_to_close", 0)
                            return True, f"Outside trading window (local={local_time}, close_in={minutes_to_close}min)"
        except Exception:
            pass
        return False, ""

    def _check_final_exit_window(self) -> Tuple[bool, bool, float]:
        try:
            if self._smart_bus:
                seasonality = self._smart_bus.get("SeasonalityRiskExpert_voting_proposal", "SmartPositionManager")
                if isinstance(seasonality, dict):
                    trading_window = seasonality.get("trading_window", {})
                    if isinstance(trading_window, dict):
                        final_exit = bool(trading_window.get("final_exit_window", False))
                        obligatory = bool(trading_window.get("final_exit_obligatory", True))
                        max_loss_pct = float(trading_window.get("final_exit_max_loss_pct", 0.02) or 0.02)
                        return final_exit, obligatory, max_loss_pct
        except Exception:
            pass
        return False, True, 0.02

    def _get_current_balance(self) -> float:
        try:
            if self._smart_bus:
                live_status = self._smart_bus.get("live_adapter_status", "SmartPositionManager", default=None)
                if isinstance(live_status, dict):
                    for key in ("equity", "balance"):
                        val = live_status.get(key)
                        if isinstance(val, (int, float)) and float(val) > 0:
                            return float(val)
        except Exception:
            pass
        return 100_000.0

    # =========================================================
    # STATE PERSISTENCE (PEAKS & INITIAL RISK)
    # =========================================================

    def _load_peaks_from_bus(self) -> Dict[str, float]:
        try:
            if self._smart_bus:
                peaks = self._smart_bus.get("smart_position_peaks", "SmartPositionManager")
                if isinstance(peaks, dict):
                    return {k: float(v) for k, v in peaks.items()}
        except Exception:
            pass
        return {}

    def _persist_peaks_to_bus(self) -> None:
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
        try:
            if self._smart_bus:
                risks = self._smart_bus.get("smart_position_initial_risk", "SmartPositionManager")
                if isinstance(risks, dict):
                    return {k: float(v) for k, v in risks.items()}
        except Exception:
            pass
        return {}

    def _persist_initial_risk_to_bus(self) -> None:
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
        initial_risk = self._initial_risk_by_symbol.get(symbol, 0.0)
        if initial_risk <= 0:
            initial_risk = self.config.scale_down_trigger_loss_eur * 2.0
        return pnl / initial_risk if initial_risk > 0 else 0.0

    def _set_initial_risk(self, symbol: str, risk_eur: float) -> None:
        if risk_eur > 0:
            self._initial_risk_by_symbol[symbol] = float(risk_eur)
            self._persist_initial_risk_to_bus()

    # =========================================================
    # LIFECYCLE & AGGRESSION
    # =========================================================

    def _get_lifecycle_state(self, position: LivePosition) -> PositionLifecycle:
        symbol = position.symbol
        cfg = self._get_symbol_cfg(symbol)

        if self._lifecycle_states.get(symbol) == PositionLifecycle.EXIT:
            return PositionLifecycle.EXIT

        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
        peak_pnl = self._profit_peaks.get(symbol, position.unrealized_pnl)

        if position.age_hours < float(cfg.get("probe_max_age_hours", 0.5)):
            if r_mult < float(cfg.get("probe_max_profit_r", 0.3)):
                return PositionLifecycle.PROBE

        if position.age_hours < float(cfg.get("build_max_age_hours", 2.0)):
            if r_mult < float(cfg.get("build_max_profit_r", 1.0)):
                return PositionLifecycle.BUILD

        if peak_pnl > 0 and position.unrealized_pnl > 0:
            drawdown_from_peak = (peak_pnl - position.unrealized_pnl) / peak_pnl if peak_pnl > 0 else 0.0
            defend_trigger = float(cfg.get("defend_trigger_drawdown_pct", self.config.defend_trigger_drawdown_pct))
            if drawdown_from_peak >= defend_trigger:
                return PositionLifecycle.DEFEND

        ride_min_r = float(cfg.get("ride_min_profit_r", self.config.ride_min_profit_r))
        if r_mult >= ride_min_r:
            return PositionLifecycle.RIDE

        return PositionLifecycle.BUILD

    def _get_aggression_multiplier(self) -> float:
        if not self.config.dd_aggression_enabled:
            return 1.0

        try:
            if self._smart_bus:
                daily_dd = self._smart_bus.get("daily_drawdown_pct", "SmartPositionManager", default=0.0)
                equity_dd = self._smart_bus.get("equity_drawdown_pct", "SmartPositionManager", default=0.0)
                current_dd = max(float(daily_dd or 0.0), float(equity_dd or 0.0))
            else:
                current_dd = 0.0
        except Exception:
            current_dd = 0.0

        if current_dd >= self.config.severe_dd_threshold:
            return self.config.severe_aggression_mult
        if current_dd >= self.config.moderate_dd_threshold:
            return self.config.moderate_aggression_mult
        if current_dd >= self.config.mild_dd_threshold:
            return self.config.mild_aggression_mult
        return 1.0

    # =========================================================
    # STARTUP GRACE
    # =========================================================

    def _get_symbol_call_count(self, symbol: str) -> int:
        return self._decide_call_count_by_symbol.get(symbol, 0)

    def _increment_symbol_call_count(self, symbol: str) -> None:
        self._decide_call_count_by_symbol[symbol] = self._get_symbol_call_count(symbol) + 1
        self._decide_call_count += 1

    def _is_signal_valid_for_symbol(self, symbol: str) -> bool:
        cfg = self._get_symbol_cfg(symbol)
        grace_calls = int(cfg.get("startup_grace_calls", self.config.default_startup_grace_calls) or 0)
        return self._get_symbol_call_count(symbol) >= grace_calls

    # =========================================================
    # PPO DECISION AWARENESS (v5.2+)
    # =========================================================

    def _get_ppo_decision(self, symbol: str, position_side: Optional[str] = None) -> PPODecisionView:
        view = PPODecisionView()
        if self._smart_bus is None:
            return view

        if position_side is None:
            position = self._positions.get(symbol)
            if position is not None:
                position_side = "long" if position.side > 0 else "short"

        def _normalise_symbol_key(key: str) -> str:
            return str(key).replace("_", "").replace(".", "").upper()

        sym_key = _normalise_symbol_key(symbol)

        try:
            multi_decision = self._smart_bus.get("ppo_multi_decision", "SmartPositionManager", default=None)
            inst_dec = None

            if hasattr(multi_decision, "instruments"):
                inst_map = getattr(multi_decision, "instruments", {}) or {}
                for k, v in inst_map.items():
                    if _normalise_symbol_key(k) == sym_key:
                        inst_dec = v
                        break

            if inst_dec is None and isinstance(multi_decision, dict):
                inst_map = multi_decision.get("instruments") or multi_decision.get("decisions") or {}
                if isinstance(inst_map, dict):
                    inst_dec = inst_map.get(symbol)
                    if inst_dec is None:
                        for k, v in inst_map.items():
                            if _normalise_symbol_key(k) == sym_key:
                                inst_dec = v
                                break

            if inst_dec is not None:
                if not isinstance(inst_dec, dict):
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
                discrete = meta.get("discrete_action") or {}
                discrete_intent = ""
                discrete_action_id = -1
                if isinstance(discrete, dict):
                    discrete_intent = str(discrete.get("intent", "") or "").lower()
                    try:
                        discrete_action_id = int(discrete.get("action_id", -1) or -1)
                    except Exception:
                        discrete_action_id = -1
                ppo_exit = meta.get("ppo_exit", {}) or {}

                explicit_close = bool(ppo_exit.get("explicit_close", False))
                explicit_reverse = bool(ppo_exit.get("explicit_reverse", False))

                wants_flat = (
                    direction == "flat"
                    or action_intent in ("close", "no_position")
                    or explicit_close
                )

                is_reversal = False
                if position_side in ("long", "short"):
                    if explicit_reverse:
                        is_reversal = True
                    elif direction in ("long", "short"):
                        is_reversal = (position_side == "long" and direction == "short") or (position_side == "short" and direction == "long")

                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.position_size = max(0.0, min(1.0, position_size))    
                view.action_intent = action_intent
                view.discrete_intent = discrete_intent
                view.discrete_action_id = discrete_action_id
                view.explicit_close = explicit_close
                view.explicit_reverse = explicit_reverse
                view.wants_flat = wants_flat
                view.is_reversal = is_reversal

                decision_sig = f"{direction}|{action_intent}|{discrete_intent}|{explicit_close}|{explicit_reverse}"
                last_sig = self._last_ppo_decision.get(symbol, "")        
                if decision_sig != last_sig:
                    self._last_ppo_decision[symbol] = decision_sig
                    if action_intent not in ("hold", "no_position", "") or explicit_close or explicit_reverse:
                        self.logger.info(
                            f"[PPO_DECISION] {symbol}: dir={view.direction} conf={view.confidence:.3f} "
                            f"intent={view.action_intent or '-'} close={view.explicit_close} reverse={view.explicit_reverse}"
                        )
                return view

            legacy_dec = self._smart_bus.get("ppo_decision", "SmartPositionManager", default=None)
            if isinstance(legacy_dec, dict):
                direction = str(legacy_dec.get("direction", "flat") or "flat").lower()
                confidence = float(legacy_dec.get("confidence", 0.0) or 0.0)
                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.wants_flat = direction == "flat"
                if position_side in ("long", "short") and direction in ("long", "short"):
                    view.is_reversal = (position_side == "long" and direction == "short") or (position_side == "short" and direction == "long")
                return view

            final_dec = self._smart_bus.get("ppo_final_decision", "SmartPositionManager", default=None)
            if isinstance(final_dec, dict):
                direction = str(final_dec.get("direction", "flat") or "flat").lower()
                confidence = float(final_dec.get("confidence", 0.0) or 0.0)
                view.direction = direction
                view.confidence = max(0.0, min(1.0, confidence))
                view.wants_flat = direction == "flat"
                if position_side in ("long", "short") and direction in ("long", "short"):
                    view.is_reversal = (position_side == "long" and direction == "short") or (position_side == "short" and direction == "long")
                return view

        except Exception as e:
            self.logger.warning(f"[PPO_DECISION] Failed to read PPO decision for {symbol}: {e}")

        return view

    def _should_respect_ppo_reversal(self, symbol: str, ppo_confidence: float, position_pnl: float) -> Tuple[bool, str]:
        if position_pnl < 0:
            reversal_threshold = 0.60
        else:
            reversal_threshold = 0.70

        if ppo_confidence >= reversal_threshold:
            return True, (
                f"PPO MASTER REVERSAL: confidence {ppo_confidence:.0%} >= "
                f"{reversal_threshold:.0%} threshold (P&L €{position_pnl:.2f})"
            )
        return False, f"PPO confidence {ppo_confidence:.0%} < {reversal_threshold:.0%}"

    def _maybe_apply_ppo_exit_override(
        self,
        symbol: str,
        position: LivePosition,
        reasons: List[str],
        support_ratio: Optional[float] = None,
        management_ctx: Optional[Dict[str, Any]] = None,
    ) -> Optional[SmartDecision]:
        """Centralized PPO explicit-close / reversal handling."""
        side_str = "long" if position.side > 0 else "short"
        ppo_view = self._get_ppo_decision(symbol, side_str)

        sym_cfg = self._get_symbol_cfg(symbol)
        ppo_exit_threshold = float(sym_cfg.get("ppo_exit_conf_threshold", self.config.ppo_exit_conf_threshold))

        # For MaskablePPO discrete actions, treat explicit CLOSE as authoritative even if
        # the derived confidence is low (flat score). For continuous models, keep the
        # confidence threshold to avoid churn from noisy near-zero outputs.
        if ppo_view.explicit_close and (
            ppo_view.discrete_intent == "close" or ppo_view.confidence >= ppo_exit_threshold
        ):
            reasons.append(f"🎯 PPO explicit CLOSE intent (conf={ppo_view.confidence:.2f})")
            self.logger.info(f"[PPO_EXIT] {symbol}: PPO explicit CLOSE (conf={ppo_view.confidence:.3f})")
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=max(0.75, ppo_view.confidence),
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
            )

        if ppo_view.is_reversal and ppo_view.confidence >= ppo_exit_threshold:
            should_respect, ppo_reason = self._should_respect_ppo_reversal(symbol, ppo_view.confidence, position.unrealized_pnl)
            if should_respect:
                target_side = int(getattr(ppo_view, "direction_int", 0) or 0)

                # If PPO wants to reverse but we can't infer a target side, fall back to a safe close.
                if target_side == 0:
                    reasons.append(f"🎯 {ppo_reason} (no target side; closing)")
                    self.logger.info(
                        f"[PPO_EXIT] {symbol}: PPO reversal requested but target_side=0; closing only"
                    )
                    return self._make_decision(
                        action=PositionAction.CLOSE,
                        symbol=symbol,
                        side=position.side,
                        confidence=max(0.70, ppo_view.confidence),
                        reasons=reasons,
                        management_context=management_ctx,
                        expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
                    )

                # Respect trading window: allow close, but block opening a new reversed position.
                no_new_trades, no_new_reason = self._check_no_new_trades()
                if no_new_trades:
                    reasons.append(f"⚠️ PPO reversal blocked: {no_new_reason} (closing only)")
                    return self._make_decision(
                        action=PositionAction.CLOSE,
                        symbol=symbol,
                        side=position.side,
                        confidence=max(0.70, ppo_view.confidence),
                        reasons=reasons,
                        management_context=management_ctx,
                        expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
                    )

                # Respect reversal cooldown: if too soon, close only (avoid flip-flop).
                last_trade = float(self._last_trade_time.get(symbol, 0.0) or 0.0)
                if last_trade > 0.0:
                    elapsed = time.time() - last_trade
                    if elapsed < self.config.reversal_cooldown_seconds:
                        remaining = max(0.0, self.config.reversal_cooldown_seconds - elapsed)
                        reasons.append(f"Reversal cooldown active ({remaining:.0f}s remaining) - closing only")
                        return self._make_decision(
                            action=PositionAction.CLOSE,
                            symbol=symbol,
                            side=position.side,
                            confidence=max(0.70, ppo_view.confidence),
                            reasons=reasons,
                            management_context=management_ctx,
                            expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
                        )

                reasons.append(f"🎯 {ppo_reason}")
                self.logger.info(
                    f"[PPO_EXIT] {symbol}: PPO reversal {side_str} → {ppo_view.direction} (conf={ppo_view.confidence:.3f})"
                )
                return self._make_decision(
                    action=PositionAction.REVERSE,
                    symbol=symbol,
                    side=target_side,
                    lots=self.config.default_lot_size,
                    confidence=max(0.70, ppo_view.confidence),
                    reasons=reasons,
                    close_first=True,
                    management_context=management_ctx,
                    expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
                )

        return None

    # =========================================================
    # LOT CALCULATOR
    # =========================================================

    @property
    def lot_calculator(self):
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
        self._positions.clear()
        self._actual_mt5_position_count = len(mt5_positions) if mt5_positions else 0

        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for pos in mt5_positions or []:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol.setdefault(symbol, []).append(pos)

        for symbol, positions in by_symbol.items():
            net_pos = self._calculate_net_position(symbol, positions)
            if net_pos and abs(net_pos.lots) > 0.001:
                self._positions[symbol] = net_pos
                prev_peak = self._profit_peaks.get(symbol, net_pos.unrealized_pnl)
                self._profit_peaks[symbol] = max(prev_peak, net_pos.unrealized_pnl)

        # Cleanup peaks for closed symbols + reset ExitEngine peak
        closed_symbols = set(self._profit_peaks.keys()) - set(self._positions.keys())
        if closed_symbols:
            exit_engine = get_exit_engine()
            for symbol in closed_symbols:
                self._profit_peaks.pop(symbol, None)
                try:
                    exit_engine.reset_peak(symbol)
                except Exception:
                    pass

        self._persist_peaks_to_bus()
        self._last_sync_time = time.time()
        return self._positions.copy()

    def _calculate_net_position(self, symbol: str, positions: List[Dict[str, Any]]) -> Optional[LivePosition]:
        if not positions:
            return None

        buy_lots = sell_lots = 0.0
        buy_value = sell_value = 0.0
        total_pnl = 0.0
        earliest_time = float("inf")
        current_price = 0.0

        # Track the largest-ticket per side (for representative SL/TP)
        best_buy: Tuple[float, int, float, float] = (0.0, 0, 0.0, 0.0)   # (lots, ticket, sl, tp)
        best_sell: Tuple[float, int, float, float] = (0.0, 0, 0.0, 0.0)

        for pos in positions:
            lots = float(pos.get("volume", pos.get("lots", 0.0)) or 0.0)
            entry = float(pos.get("price_open", pos.get("entry_price", 0.0)) or 0.0)
            pnl = float(pos.get("profit", pos.get("unrealized_pnl", 0.0)) or 0.0)
            open_time = float(pos.get("time", pos.get("open_time", 0.0)) or 0.0)
            pos_type = pos.get("type", 0)
            ticket = int(pos.get("ticket", 0) or 0)
            sl = float(pos.get("sl", 0.0) or 0.0)
            tp = float(pos.get("tp", 0.0) or 0.0)

            is_buy = (pos_type == 0)

            if is_buy:
                buy_lots += lots
                buy_value += lots * entry
                if lots > best_buy[0]:
                    best_buy = (lots, ticket, sl, tp)
            else:
                sell_lots += lots
                sell_value += lots * entry
                if lots > best_sell[0]:
                    best_sell = (lots, ticket, sl, tp)

            total_pnl += pnl

            if open_time > 0:
                earliest_time = min(earliest_time, open_time)

            if current_price == 0.0:
                current_price = float(pos.get("price_current", entry) or entry)

        net_lots = buy_lots - sell_lots
        if abs(net_lots) < 0.001:
            return None

        side = 1 if net_lots > 0 else -1
        abs_lots = abs(net_lots)

        # VWAP entry for net side
        if side > 0 and buy_lots > 0:
            entry_price = buy_value / buy_lots
        elif side < 0 and sell_lots > 0:
            entry_price = sell_value / sell_lots
        else:
            entry_price = current_price

        # Representative SL/TP should come from the dominant ticket on the NET side
        if side > 0:
            _, ticket, position_sl, position_tp = best_buy
        else:
            _, ticket, position_sl, position_tp = best_sell

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

        # Reset state for hard exits / reversals
        if dec.action in (PositionAction.CLOSE, PositionAction.REVERSE):
            self._profit_peaks.pop(dec.symbol, None)
            self._lifecycle_states.pop(dec.symbol, None)
            self._initial_risk_by_symbol.pop(dec.symbol, None)
            self._persist_initial_risk_to_bus()
            try:
                get_exit_engine().reset_peak(dec.symbol)
            except Exception:
                pass

        self._log_decision_box(dec)
        return dec

    def _log_decision_box(self, decision: SmartDecision) -> None:
        if decision.action == PositionAction.HOLD:
            return
        try:
            header = f"SMART POSITION DECISION – {decision.symbol}"
            box_width = 78

            lines_box: List[str] = []
            lines_box.append("")
            lines_box.append("┌─ " + header + " " + "─" * max(0, box_width - len(header) - 3))
            lines_box.append(f"│ Action:        {decision.action.value:<18} Side: {decision.side:+d}")
            lines_box.append(f"│ Lots:          {decision.lots:.2f}                Confidence: {decision.confidence:.2f}")

            if decision.new_sl is not None or decision.new_tp is not None:
                sl_str = f"{decision.new_sl:.5f}" if decision.new_sl is not None else "unchanged"
                tp_str = f"{decision.new_tp:.5f}" if decision.new_tp is not None else "unchanged"
                lines_box.append(f"│ New SL:        {sl_str:<18} New TP: {tp_str}")

            if decision.expert_support_ratio != 0.5:
                lines_box.append(f"│ Expert Support: {decision.expert_support_ratio:.0%}")

            if decision.close_first:
                lines_box.append("│ Note:          Close existing position before applying action")

            if decision.reasons:
                lines_box.append("│ Reasons:")
                for r in decision.reasons[:3]:
                    lines_box.append(f"│   • {r}")

            lines_box.append("└" + "─" * (box_width - 1))
            self.logger.info("\n".join(lines_box))
        except Exception:
            pass

    # =========================================================
    # SL / PROTECTION HELPERS (BREAKEVEN + PROFIT LOCK)
    # =========================================================

    def _calculate_breakeven_sl(self, position: LivePosition) -> Optional[float]:
        try:
            if position.entry_price <= 0:
                return None
            buffer = position.entry_price * 0.0002  # ~2 pips for FX, ~0.4 for XAU @ 2000
            if position.side > 0:
                return round(position.entry_price + buffer, 5)
            return round(position.entry_price - buffer, 5)
        except Exception:
            return None

    def _is_sl_improvement(self, position: LivePosition, new_sl: float) -> bool:
        try:
            if new_sl <= 0:
                return False
            if position.sl <= 0:
                return True
            if position.side > 0:
                return new_sl > position.sl
            return new_sl < position.sl
        except Exception:
            return False

    def _calculate_profit_lock_sl(
        self,
        position: LivePosition,
        lifecycle: PositionLifecycle,
        r_mult: float,
    ) -> Optional[float]:
        """
        Adaptive profit-lock ladder:
        - after breakeven is plausible, lock a fraction of open profit
        - in DEFEND, lock more aggressively to avoid "late trailing" exits
        """
        if not self.config.enable_profit_lock:
            return None

        # Gate 1: profit must be meaningful (R-based AND EUR-based fallback)
        sym_cfg = self._get_symbol_cfg(position.symbol)
        be_eur = float(sym_cfg.get("breakeven_activation_eur", self.config.breakeven_activation_eur))
        eur_gate = be_eur * float(self.config.profit_lock_min_eur_mult)
        if position.unrealized_pnl < eur_gate and r_mult < self.config.profit_lock_min_r:
            return None

        # Determine lock fraction from ladder
        lock_frac = 0.0
        for r_thr, frac in self.config.profit_lock_ladder:
            if r_mult >= float(r_thr):
                lock_frac = max(lock_frac, float(frac))

        if lock_frac <= 0.0:
            return None

        # DEFEND mode: tighten more
        if lifecycle == PositionLifecycle.DEFEND:
            lock_frac = min(0.80, lock_frac + 0.12)

        # Compute price-based profit and lock fraction of it
        if position.current_price <= 0 or position.entry_price <= 0:
            return None

        open_profit_price = position.current_price - position.entry_price
        if position.side < 0:
            open_profit_price = position.entry_price - position.current_price

        if open_profit_price <= 0:
            return None

        locked_distance = open_profit_price * lock_frac

        if position.side > 0:
            new_sl = position.entry_price + locked_distance
        else:
            new_sl = position.entry_price - locked_distance

        return round(new_sl, 5)

    def _calculate_tightened_sl(self, position: LivePosition, signal: PositionManagementSignal) -> Optional[float]:
        """Fallback tightened SL (simple, price-based) when experts support holding but risk rises."""
        try:
            if position.current_price <= 0 or position.entry_price <= 0:
                return None

            profit_per_unit = position.current_price - position.entry_price
            if position.side < 0:
                profit_per_unit = position.entry_price - position.current_price

            protection_buffer = abs(profit_per_unit) * 0.5
            if position.side > 0:
                new_sl = position.current_price - protection_buffer
            else:
                new_sl = position.current_price + protection_buffer

            return round(new_sl, 5)
        except Exception:
            return None

    # =========================================================
    # FINAL EXIT WINDOW HELPER
    # =========================================================

    def _maybe_force_final_exit(
        self,
        symbol: str,
        position: LivePosition,
        reasons: List[str],
        management_ctx: Optional[Dict[str, Any]] = None,
        support_ratio: Optional[float] = None,
    ) -> Optional[SmartDecision]:
        in_final_exit, is_obligatory, max_loss_pct = self._check_final_exit_window()
        if not (in_final_exit and is_obligatory):
            return None

        balance = self._get_current_balance()
        max_loss_amount = balance * max_loss_pct

        if position.unrealized_pnl >= 0:
            reasons.append(
                f"🌙 FINAL EXIT WINDOW: Closing profitable/breakeven position (P&L €{position.unrealized_pnl:.2f})"
            )
            self.logger.info(f"[FINAL_EXIT] {symbol}: Closing position (P&L €{position.unrealized_pnl:.2f})")
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=0.80,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
            )

        if abs(position.unrealized_pnl) <= max_loss_amount:
            reasons.append(
                f"🌙 FINAL EXIT WINDOW: Closing position with acceptable loss (P&L €{position.unrealized_pnl:.2f} within €{max_loss_amount:.2f})"
            )
            self.logger.info(
                f"[FINAL_EXIT] {symbol}: Closing losing position (P&L €{position.unrealized_pnl:.2f} within €{max_loss_amount:.2f})"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=0.75,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=(support_ratio if support_ratio is not None else 0.5),
            )

        reasons.append(
            f"⚠️ FINAL EXIT WINDOW: Loss too large to force close (P&L €{position.unrealized_pnl:.2f} exceeds €{max_loss_amount:.2f}) - letting ExitEngine manage"
        )
        self.logger.warning(
            f"[FINAL_EXIT] {symbol}: Loss €{position.unrealized_pnl:.2f} exceeds max €{max_loss_amount:.2f} - not forcing close"
        )
        return None

    # =========================================================
    # PUBLIC DECISION API - SIMPLE MODE
    # =========================================================

    def decide(
        self,
        symbol: str,
        signal_direction: int,
        signal_strength: float,
        consensus_confidence: float = 0.5,
    ) -> SmartDecision:
        cfg = self.config
        sym_cfg = self._get_symbol_cfg(symbol)

        self._increment_symbol_call_count(symbol)
        signal_valid = self._is_signal_valid_for_symbol(symbol)

        # Sanitise inputs
        signal_direction = 1 if signal_direction > 0 else (-1 if signal_direction < 0 else 0)
        signal_strength = float(max(0.0, min(1.0, float(signal_strength))))
        consensus_confidence = float(max(0.0, min(1.0, float(consensus_confidence))))

        position = self._positions.get(symbol)
        reasons: List[str] = []

        # No position
        if position is None:
            return self._decide_new_position(
                symbol=symbol,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=consensus_confidence,
                reasons=reasons,
            )

        # Update local peak
        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        # Final exit window can force close
        forced = self._maybe_force_final_exit(symbol, position, reasons)
        if forced is not None:
            return forced

        lifecycle = self._get_lifecycle_state(position)
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)

        # PRIORITY -1: Breakeven + profit-lock BEFORE ExitEngine trailing close can trigger
        be_threshold = float(sym_cfg.get("breakeven_activation_eur", cfg.breakeven_activation_eur))
        if position.unrealized_pnl >= be_threshold:
            be_sl = self._calculate_breakeven_sl(position)
            if be_sl is not None and self._is_sl_improvement(position, be_sl):
                reasons.append(
                    f"🔒 BREAKEVEN FIRST: Profit €{position.unrealized_pnl:.2f} >= €{be_threshold:.0f} → moving SL to breakeven"
                )
                reasons.append(
                    f"PROFIT LOCK (decide): {r_mult:.1f}R / €{position.unrealized_pnl:.0f} → tighten SL to lock gains"
                )
                return self._make_decision(
                    action=PositionAction.ADJUST_SL,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                    new_sl=be_sl,
                )

            lock_sl = self._calculate_profit_lock_sl(position, lifecycle, r_mult)
            if lock_sl is not None and self._is_sl_improvement(position, lock_sl):
                reasons.append(
                    f"🔒 PROFIT LOCK: {r_mult:.1f}R / €{position.unrealized_pnl:.0f} → tightening SL to lock gains"
                )
                return self._make_decision(
                    action=PositionAction.TIGHTEN_PROTECTION,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                    new_sl=lock_sl,
                )

        # ExitEngine evaluation
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
            regime=lifecycle.value if lifecycle else "normal",
        )

        exit_decision = get_exit_engine().evaluate(exit_ctx)
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

        # Align/oppose checks
        signal_aligns = (position.side > 0 and signal_direction > 0) or (position.side < 0 and signal_direction < 0)
        signal_against = (position.side > 0 and signal_direction < 0) or (position.side < 0 and signal_direction > 0)

        # Aligning -> scale logic
        if signal_aligns and signal_direction != 0:
            return self._decide_scale(
                symbol=symbol,
                position=position,
                signal_strength=signal_strength,
                consensus_confidence=consensus_confidence,
                reasons=reasons,
            )

        # PPO override even if ExitEngine said HOLD
        ppo_override = self._maybe_apply_ppo_exit_override(symbol, position, reasons)
        if ppo_override is not None:
            return ppo_override

        # Strong opposing signal -> reversal/close logic
        if signal_against and signal_strength >= cfg.strong_signal_threshold:
            reasons.append(f"REVERSAL: Strong opposing signal ({signal_strength:.2f}) vs {position.direction} position")

            if position.unrealized_pnl < 0:
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                )

            last_trade = self._last_trade_time.get(symbol, 0.0)
            elapsed = time.time() - last_trade
            if elapsed < cfg.reversal_cooldown_seconds:
                remaining = max(0.0, cfg.reversal_cooldown_seconds - elapsed)
                reasons.append(f"Reversal cooldown active ({remaining:.0f}s remaining) – skipping flip")
                return self._make_decision(
                    action=PositionAction.HOLD,
                    symbol=symbol,
                    confidence=0.65,
                    reasons=reasons,
                )

            no_new_trades, no_new_reason = self._check_no_new_trades()
            if no_new_trades:
                reasons.append(f"🚫 REVERSE BLOCKED: {no_new_reason} - closing instead")
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                )

            return self._make_decision(
                action=PositionAction.REVERSE,
                symbol=symbol,
                lots=self.config.default_lot_size,
                side=signal_direction,
                confidence=0.70,
                reasons=reasons,
                close_first=True,
            )

        reasons.append(
            f"HOLD: {position.direction} {position.lots:.2f} lots, P&L €{position.unrealized_pnl:.2f}, age {position.age_hours:.1f}h"
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
        cfg = self.config
        sym_cfg = self._get_symbol_cfg(symbol)

        no_new_trades, no_new_reason = self._check_no_new_trades()
        if no_new_trades:
            reasons.append(f"🚫 BLOCKED: {no_new_reason}")
            self.logger.info(f"[TRADING_WINDOW] {symbol}: Blocked new position - {no_new_reason}")
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        existing = self._positions.get(symbol)
        if existing is not None:
            if (existing.side > 0 and signal_direction < 0) or (existing.side < 0 and signal_direction > 0):
                reasons.append(f"BLOCKED: Existing {existing.direction} position exists, won't open opposing position")
            else:
                reasons.append(f"BLOCKED: Already have {existing.direction} position, prefer SCALE instead of new OPEN")
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        min_strength = float(sym_cfg.get("min_signal_strength", cfg.min_signal_strength))
        if signal_strength < min_strength:
            reasons.append(f"Signal too weak ({signal_strength:.2f} < {min_strength:.2f})")
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        if signal_direction == 0:
            reasons.append("Neutral signal - no direction")
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        last_trade = self._last_trade_time.get(symbol, 0.0)
        cooldown_remaining = cfg.same_direction_cooldown_seconds - (time.time() - last_trade)
        if cooldown_remaining > 0:
            reasons.append(f"Cooldown active ({cooldown_remaining:.0f}s remaining)")
            self._publish_cooldown_state(symbol, True, float(max(cooldown_remaining, 0.0)), last_trade, cfg.same_direction_cooldown_seconds)
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        effective_count = max(len(self._positions), self._actual_mt5_position_count)
        if effective_count >= cfg.max_total_positions:
            reasons.append(
                f"Max positions reached (net={len(self._positions)}, actual={self._actual_mt5_position_count}, max={cfg.max_total_positions})"
            )
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.5, reasons=reasons)

        action = PositionAction.OPEN_LONG if signal_direction > 0 else PositionAction.OPEN_SHORT

        lots = 0.0
        initial_risk_eur = 0.0
        if self.lot_calculator and cfg.use_unified_calculator:
            try:
                lots, lot_details = self.lot_calculator.calculate_lots(symbol=symbol, signal_strength=signal_strength)
                initial_risk_eur = float(
                    lot_details.get("risk_eur", lot_details.get("risk_pct", 0.005) * lot_details.get("balance", 100000))
                )
                self.logger.info(
                    f"[SMART_PM] 📊 UNIFIED_LOT: {symbol} | signal={signal_strength:.2f} | lots={lots:.2f} | "
                    f"balance=€{lot_details.get('balance', 0):.0f} | risk={lot_details.get('risk_pct', 0)*100:.1f}% | "
                    f"initial_risk_eur=€{initial_risk_eur:.2f}"
                )
            except Exception as e:
                self.logger.warning(f"[SMART_PM] Unified lot calc failed: {e}, using fallback")
                lots = 0.0

        if lots <= 0:
            raw_lots = cfg.default_lot_size * signal_strength
            lots = min(max(raw_lots, 0.01), cfg.max_lot_size)
            initial_risk_eur = 50.0

        self._set_initial_risk(symbol, initial_risk_eur)
        self._lifecycle_states[symbol] = PositionLifecycle.PROBE

        reasons.append(
            f"OPEN {action.value}: signal={signal_strength:.2f}, consensus={consensus_confidence:.2f}, lots={lots:.2f}, initial_risk=€{initial_risk_eur:.2f}"
        )
        return self._make_decision(
            action=action,
            symbol=symbol,
            lots=round(lots, 2),
            side=signal_direction,
            confidence=min(signal_strength, consensus_confidence),
            reasons=reasons,
        )

    def _publish_cooldown_state(self, symbol: str, on_cooldown: bool, remaining: float, last_trade_ts: float, cooldown_seconds: float) -> None:
        try:
            if self._smart_bus is None:
                return
            cooldown_state = self._smart_bus.get("instrument_cooldown_state", "SmartPositionManager", default={}) or {}
            if not isinstance(cooldown_state, dict):
                cooldown_state = {}
            sym_state = cooldown_state.get(symbol, {})
            if not isinstance(sym_state, dict):
                sym_state = {}
            sym_state.update(
                {
                    "on_cooldown": bool(on_cooldown),
                    "cooldown_remaining": float(max(remaining, 0.0)),
                    "last_trade_ts": float(last_trade_ts),
                    "cooldown_seconds": float(cooldown_seconds),
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
            pass

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
        cfg = self.config

        if not cfg.enable_scale_up:
            reasons.append("Scale-up disabled (enable_scale_up=false) – enforcing single-ticket behavior")
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.60, reasons=reasons)

        lifecycle = self._get_lifecycle_state(position)
        if lifecycle == PositionLifecycle.DEFEND:
            r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
            peak_pnl = self._profit_peaks.get(symbol, position.unrealized_pnl)
            retrace_pct = (peak_pnl - position.unrealized_pnl) / peak_pnl if peak_pnl > 0 else 0.0
            reasons.append(
                f"🛡️ DEFEND MODE: Profit retracing ({retrace_pct:.0%} from peak €{peak_pnl:.2f}), blocking scale-up. "
                f"Current €{position.unrealized_pnl:.2f} ({r_mult:.1f}R)"
            )
            return self._make_decision(PositionAction.HOLD, symbol, confidence=0.60, reasons=reasons)

        last_scale = self._last_scale_time.get(symbol, 0.0)
        scale_cooldown_ok = (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds

        sym_cfg = self._get_symbol_cfg(symbol)
        strong_threshold = float(sym_cfg.get("strong_signal_threshold", cfg.strong_signal_threshold))

        use_r_based = cfg.use_r_based_scaling
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
        scale_up_min_r = float(sym_cfg.get("scale_up_min_r", cfg.scale_up_min_r))
        scale_up_min_profit = float(sym_cfg.get("scale_up_min_profit_eur", cfg.scale_up_min_profit_eur))

        profit_ok = (use_r_based and r_mult >= scale_up_min_r) or (not use_r_based and position.unrealized_pnl >= scale_up_min_profit)

        if (
            signal_strength >= strong_threshold
            and profit_ok
            and scale_cooldown_ok
            and position.lots < cfg.max_lot_size
        ):
            add_lots = 0.0
            if self.lot_calculator and cfg.use_unified_calculator:
                try:
                    aggression = self._get_aggression_multiplier()
                    scale_signal = signal_strength * 0.5 * aggression
                    add_lots, _ = self.lot_calculator.calculate_lots(symbol=symbol, signal_strength=scale_signal)
                    add_lots = min(add_lots, cfg.max_lot_size - position.lots)
                except Exception:
                    add_lots = 0.0

            if add_lots <= 0:
                add_lots = min(cfg.default_lot_size * 0.5, cfg.max_lot_size - position.lots)

            if add_lots >= 0.01:
                reasons.append(
                    f"📈 SCALE UP [{lifecycle.value.upper()}]: +€{position.unrealized_pnl:.2f} profit ({r_mult:.1f}R), "
                    f"strong signal ({signal_strength:.2f})"
                )
                return self._make_decision(
                    action=PositionAction.SCALE_UP,
                    symbol=symbol,
                    lots=round(add_lots, 2),
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                )

        # Optional smart scale-down for losers
        scale_result = self._smart_scale_down_common(position=position, expert_signal_strength=signal_strength, support_ratio=None)
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

        reasons.append(
            f"HOLD [{lifecycle.value.upper()}] (aligning): {position.direction} {position.lots:.2f} lots, "
            f"P&L €{position.unrealized_pnl:.2f} ({r_mult:.1f}R)"
        )
        return self._make_decision(PositionAction.HOLD, symbol, confidence=0.60, reasons=reasons)

    # =========================================================
    # SMART SCALE DOWN (OPTIONAL, LOSING POSITIONS)
    # =========================================================

    def _smart_scale_down_common(
        self,
        position: LivePosition,
        expert_signal_strength: float,
        support_ratio: Optional[float] = None,
    ) -> Optional[Tuple[PositionAction, float, str]]:
        if not getattr(self.config, "enable_smart_scale_down", False):
            return None

        symbol = position.symbol
        cfg = self._get_symbol_cfg(symbol)

        if position.lots <= 0.01 or position.unrealized_pnl >= 0:
            return None

        lifecycle = self._get_lifecycle_state(position)
        aggression = self._get_aggression_multiplier()

        use_r_based = self.config.use_r_based_scaling

        if use_r_based:
            r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)
            loss_metric = r_mult
            trigger_threshold = float(cfg.get("scale_down_trigger_r", self.config.scale_down_trigger_r))
            aggressive_threshold = float(cfg.get("scale_down_aggressive_r", self.config.scale_down_aggressive_r))
            emergency_threshold = float(cfg.get("scale_down_emergency_r", self.config.scale_down_emergency_r))
            metric_name = "R"
        else:
            loss_amount = min(0.0, position.unrealized_pnl)
            trigger_loss = float(cfg.get("scale_down_trigger_loss_eur", self.config.scale_down_trigger_loss_eur))
            loss_metric = loss_amount
            trigger_threshold = -trigger_loss
            aggressive_threshold = -trigger_loss * 1.5
            emergency_threshold = -trigger_loss * 2.0
            metric_name = "EUR"

        conviction = support_ratio if support_ratio is not None else expert_signal_strength
        expert_weakening = conviction < 0.40
        expert_uncertain = conviction < 0.55

        if lifecycle == PositionLifecycle.PROBE:
            trigger_threshold *= 0.7
        elif lifecycle == PositionLifecycle.RIDE:
            trigger_threshold *= 1.3
        elif lifecycle == PositionLifecycle.DEFEND:
            trigger_threshold *= 0.8

        should_scale_down = False
        reduce_pct = 0.0
        scale_reason = ""

        if loss_metric > trigger_threshold:
            should_scale_down = False
        elif loss_metric > aggressive_threshold:
            if expert_weakening:
                should_scale_down = True
                reduce_pct = 0.3 * aggression
                scale_reason = f"experts weakening ({conviction:.2f})"
            elif expert_uncertain and loss_metric <= (trigger_threshold + aggressive_threshold) / 2.0:
                should_scale_down = True
                reduce_pct = 0.4 * aggression
                scale_reason = f"experts uncertain ({conviction:.2f}) + growing loss"
        elif loss_metric > emergency_threshold:
            should_scale_down = True
            reduce_pct = 0.5 * aggression
            scale_reason = f"aggressive zone + weak experts ({conviction:.2f})" if expert_weakening else f"aggressive zone, loss={loss_metric:.2f}{metric_name}"
        else:
            should_scale_down = True
            reduce_pct = 0.6
            scale_reason = f"EMERGENCY: loss={loss_metric:.2f}{metric_name} <= emergency threshold"

        if should_scale_down and reduce_pct > 0.0:
            reduce_lots = min(position.lots * reduce_pct, position.lots - 0.01)
            if reduce_lots >= 0.01:
                lifecycle_tag = f"[{lifecycle.value.upper()}]" if lifecycle != PositionLifecycle.BUILD else ""
                full_reason = (
                    f"🛡️ SMART SCALE DOWN {lifecycle_tag}: {loss_metric:.2f}{metric_name} loss, "
                    f"{scale_reason}, reducing {reduce_pct*100:.0f}%"
                )
                return (PositionAction.SCALE_DOWN, round(reduce_lots, 2), full_reason)

        return None

    # =========================================================
    # COOPERATIVE MANAGEMENT MODE (manage_position)
    # =========================================================

    def manage_position(self, signal: PositionManagementSignal) -> SmartDecision:
        symbol = signal.symbol
        position = self._positions.get(symbol)
        cfg = self.config
        sym_cfg = self._get_symbol_cfg(symbol)
        reasons: List[str] = []

        if position is None:
            direction = 1 if signal.consensus_action == "BUY" else (-1 if signal.consensus_action == "SELL" else 0)
            return self.decide(
                symbol=symbol,
                signal_direction=direction,
                signal_strength=signal.consensus_confidence,
                consensus_confidence=signal.consensus_score,
            )

        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        lifecycle = self._get_lifecycle_state(position)
        r_mult = self._get_r_multiple(symbol, position.unrealized_pnl)

        # Expert alignment
        position_action = "BUY" if position.side > 0 else "SELL"
        supporting_experts: List[ExpertSignal] = []
        opposing_experts: List[ExpertSignal] = []

        for expert in signal.expert_signals:
            if expert.action == position_action or expert.action == "HOLD":
                supporting_experts.append(expert)
            else:
                opposing_experts.append(expert)

        total_experts = len(signal.expert_signals)
        support_ratio = (len(supporting_experts) / total_experts) if total_experts > 0 else 0.5

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

        forced = self._maybe_force_final_exit(symbol, position, reasons, management_ctx=management_ctx, support_ratio=support_ratio)
        if forced is not None:
            return forced

        # PPO explicit exits/reversals (if any) should override management heuristics.
        ppo_override = self._maybe_apply_ppo_exit_override(
            symbol,
            position,
            reasons,
            support_ratio=support_ratio,
            management_ctx=management_ctx,
        )
        if ppo_override is not None:
            return ppo_override

        # PRIORITY -1: Breakeven + Profit-lock ladder
        be_threshold = float(sym_cfg.get("breakeven_activation_eur", cfg.breakeven_activation_eur))
        if position.unrealized_pnl >= be_threshold:
            be_sl = self._calculate_breakeven_sl(position)
            if be_sl is not None and self._is_sl_improvement(position, be_sl):
                reasons.append(
                    f"🔒 BREAKEVEN FIRST: Profit €{position.unrealized_pnl:.2f} >= €{be_threshold:.0f} → moving SL to breakeven"
                )
                return self._make_decision(
                    action=PositionAction.ADJUST_SL,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.75,
                    reasons=reasons,
                    new_sl=be_sl,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            lock_sl = self._calculate_profit_lock_sl(position, lifecycle, r_mult)
            if lock_sl is not None and self._is_sl_improvement(position, lock_sl):
                reasons.append(f"🔒 PROFIT LOCK: {r_mult:.1f}R → tightening SL to lock gains (lifecycle={lifecycle.value})")
                return self._make_decision(
                    action=PositionAction.TIGHTEN_PROTECTION,
                    symbol=symbol,
                    side=position.side,
                    confidence=0.70,
                    reasons=reasons,
                    new_sl=lock_sl,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

        # ExitEngine
        signal_dir = 1 if signal.consensus_action == "BUY" else (-1 if signal.consensus_action == "SELL" else 0)
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

        exit_decision = get_exit_engine().evaluate(exit_ctx)
        management_ctx.update(
            {
                "exit_engine_reason": exit_decision.reason.name,
                "exit_engine_should_exit": exit_decision.should_exit,
                "exit_engine_confidence": exit_decision.confidence,
                "lifecycle_state": lifecycle.value,
                "r_multiple": r_mult,
            }
        )

        if exit_decision.should_exit:
            engine_msg = exit_decision.details.get("message", exit_decision.reason.name)
            reasons.append(f"[ExitEngine] {engine_msg}")

            if exit_decision.is_critical:
                reasons.append("Critical exit signalled by ExitEngine; overriding expert opinions.")
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,
                    confidence=exit_decision.confidence,
                    reasons=reasons,
                    management_context=management_ctx,
                    expert_support_ratio=support_ratio,
                )

            # PPO override opportunity
            ppo_override = self._maybe_apply_ppo_exit_override(symbol, position, reasons, support_ratio=support_ratio, management_ctx=management_ctx)
            if ppo_override is not None:
                return ppo_override

            if support_ratio <= 0.3 or position.unrealized_pnl <= 0:
                reasons.append(f"Non-critical exit aligned with experts ({int((1.0 - support_ratio) * 100)}% against or flat/losing).")
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
                reasons.append("ExitEngine suggests non-critical exit, but experts support the position and it is profitable – tightening SL instead.")
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

            reasons.append("Unable to compute a better protective SL – respecting ExitEngine non-critical exit.")
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=exit_decision.confidence,
                reasons=reasons,
                management_context=management_ctx,
                expert_support_ratio=support_ratio,
            )

        # ExitEngine HOLD: PPO override still allowed
        ppo_override = self._maybe_apply_ppo_exit_override(symbol, position, reasons, support_ratio=support_ratio, management_ctx=management_ctx)
        if ppo_override is not None:
            return ppo_override

        # Scale-up when strongly supported + sufficiently profitable (still gated by enable_scale_up)
        scale_up_min_profit = float(sym_cfg.get("scale_up_min_profit_eur", cfg.scale_up_min_profit_eur))
        scale_up_min_r = float(sym_cfg.get("scale_up_min_r", cfg.scale_up_min_r))
        use_r_based = bool(cfg.use_r_based_scaling)
        profit_ok_for_scale = ((use_r_based and r_mult >= scale_up_min_r) or (not use_r_based and position.unrealized_pnl >= scale_up_min_profit))

        if support_ratio >= 0.6 and position.unrealized_pnl > 0:
            if cfg.enable_scale_up and support_ratio >= 0.75 and profit_ok_for_scale:
                last_scale = self._last_scale_time.get(symbol, 0.0)
                if (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds and position.lots < cfg.max_lot_size:
                    add_lots = self._calculate_scale_lots(position, signal)
                    if add_lots >= 0.01:
                        reasons.append(
                            f"📈 SCALE UP: {int(support_ratio * 100)}% support, +€{position.unrealized_pnl:.2f} ({r_mult:.1f}R), "
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

            # If profit is very high, tighten further (beyond ladder, fallback tightening)
            if position.unrealized_pnl >= be_threshold * 1.8:
                new_sl = self._calculate_tightened_sl(position, signal)
                if new_sl is not None and self._is_sl_improvement(position, new_sl):
                    reasons.append(
                        f"🔒 TIGHTEN FURTHER: High profit €{position.unrealized_pnl:.2f} ({int(support_ratio * 100)}% support)"
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

        # Optional smart scale-down
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

        reasons.append(
            f"✓ HOLD [{lifecycle.value.upper()}]: {position.direction} {position.lots:.2f} lots | "
            f"PnL €{position.unrealized_pnl:.2f} ({r_mult:.1f}R) | {int(support_ratio * 100)}% support | age {position.age_hours:.1f}h"
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
    # SCALE LOTS HELPER
    # =========================================================

    def _calculate_scale_lots(self, position: LivePosition, signal: PositionManagementSignal) -> float:
        try:
            cfg = self.config
            if self.lot_calculator and cfg.use_unified_calculator:
                scale_signal = float(signal.consensus_confidence) * float(signal.support_ratio) * 0.5
                add_lots, _ = self.lot_calculator.calculate_lots(symbol=position.symbol, signal_strength=scale_signal)
            else:
                add_lots = cfg.default_lot_size * float(signal.support_ratio) * 0.5

            max_add = cfg.max_lot_size - position.lots
            add_lots = min(add_lots, max_add, cfg.default_lot_size)
            return round(max(0.01, float(add_lots)), 2)
        except Exception:
            return 0.0

    # =========================================================
    # PUBLIC UTILS
    # =========================================================

    def record_trade(self, symbol: str, is_scale: bool = False) -> None:
        now = time.time()
        self._last_trade_time[symbol] = now
        if is_scale:
            self._last_scale_time[symbol] = now

    def get_position(self, symbol: str) -> Optional[LivePosition]:
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, LivePosition]:
        return self._positions.copy()

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions

    def get_net_exposure(self) -> Dict[str, float]:
        return {symbol: pos.lots * pos.side for symbol, pos in self._positions.items()}

    def get_total_pnl(self) -> float:
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    def needs_hedge_cleanup(self, mt5_positions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        to_close: List[Dict[str, Any]] = []
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}

        for pos in mt5_positions or []:
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
                        format_operator_message("🔄", "HEDGE_CLEANUP", symbol=symbol, action="Closing BUY side", buy_lots=f"{buy_lots:.2f}", sell_lots=f"{sell_lots:.2f}")
                    )
                else:
                    to_close.extend(sells)
                    self.logger.warning(
                        format_operator_message("🔄", "HEDGE_CLEANUP", symbol=symbol, action="Closing SELL side", buy_lots=f"{buy_lots:.2f}", sell_lots=f"{sell_lots:.2f}")
                    )

        return to_close

    def needs_consolidation(self, mt5_positions: List[Dict[str, Any]]) -> bool:
        by_symbol: Dict[str, int] = {}
        for pos in mt5_positions or []:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol[symbol] = by_symbol.get(symbol, 0) + 1

        return any(count > self.config.max_positions_per_symbol for count in by_symbol.values())

    def get_actual_position_count(self) -> int:
        return self._actual_mt5_position_count

    def has_hedged_positions(self) -> bool:
        return self._actual_mt5_position_count > len(self._positions)

    def log_status(self) -> None:
        if not self._positions:
            self.logger.info("📊 No open positions")
            return

        total_pnl = self.get_total_pnl()
        self.logger.info(format_operator_message("📊", "POSITION_STATUS", count=len(self._positions), total_pnl=f"€{total_pnl:.2f}"))

        for symbol, pos in self._positions.items():
            self.logger.info(
                f"  {symbol}: {pos.direction} {pos.lots:.2f} lots @ {pos.entry_price:.5f} | "
                f"P&L: €{pos.unrealized_pnl:.2f} | Age: {pos.age_hours:.1f}h | SL: {pos.sl:.5f} | TP: {pos.tp:.5f}"
            )
