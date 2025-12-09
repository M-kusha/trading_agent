# -------------------------------------------------------------
# File: modules/position/smart_position_manager.py
# Smart Position Management - Professional Net Position System
#
# Design Principles:
#   1. ONE position per symbol (net position approach)
#   2. No hedging (BUY+SELL same symbol = waste of spread)
#   3. Signal reversal = CLOSE existing, then open new direction
#   4. Smart scaling: add to winners, cut losers early
#   5. Time-aware exits: aging positions get scrutinized
#
# EXIT LOGIC: Uses unified ExitStrategyEngine (single source of truth)
# This ensures consistency with PositionManager (training) and eliminates
# duplicate exit code. See exit_engine.py for all exit strategy logic.
# -------------------------------------------------------------

from __future__ import annotations

import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from modules.utils.audit_utils import RotatingLogger, format_operator_message

# UNIFIED EXIT LOGIC - Single source of truth for all exit decisions
from .exit_engine import (
    ExitStrategyEngine,
    ExitDecision,
    ExitReason,
    PositionContext,
    get_exit_engine,
)


def load_config_from_yaml() -> Dict[str, Any]:
    """Load smart position config from risk_policy.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}
            return policy.get("smart_position", {})
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
            return policy.get("lot_sizing", {})
    except Exception as e:
        print(f"[SmartPositionManager] Failed to load lot config: {e}")
    return {}


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
    # Key: instrument symbol, Value: position details
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
                ctx.primary_age_hours = (_time.time() - float(open_time)) / 3600.0

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
    Configuration for smart position management.

    NOTE: Exit thresholds are now managed by ExitConfig (exit_engine.py).
    The fields below marked as DEPRECATED are kept for backward compatibility
    but are NOT used - ExitStrategyEngine.evaluate() uses risk_policy.yaml values.

    NOTE: Lot sizing uses UnifiedLotCalculator as the single source of truth.
    """
    # Position limits
    max_positions_per_symbol: int = 1
    max_total_positions: int = 4

    # Lot sizing - these are FALLBACKS, actual sizing uses UnifiedLotCalculator
    default_lot_size: float = 0.01     # Fallback only - calculator determines actual size
    max_lot_size: float = 50.0         # From unified lot_sizing config
    use_unified_calculator: bool = True  # Use UnifiedLotCalculator for lot sizing

    # DEPRECATED: Exit thresholds - now in ExitConfig (exit_engine.py)
    # Kept for backward compatibility but NOT USED by decide()/manage_position()
    profit_take_activation_eur: float = 50.0   # DEPRECATED - use ExitConfig
    profit_take_trail_pct: float = 0.35        # DEPRECATED - use ExitConfig
    momentum_exit_profit_eur: float = 40.0     # DEPRECATED - use ExitConfig
    hard_stop_loss_eur: float = 100.0          # DEPRECATED - use ExitConfig
    soft_stop_loss_eur: float = 30.0           # DEPRECATED - use ExitConfig
    time_decay_stop_eur: float = 50.0          # DEPRECATED - use ExitConfig
    time_decay_hours: float = 4.0              # DEPRECATED - use ExitConfig

    # Scaling rules (still used)
    scale_up_min_profit_eur: float = 10.0        # Only scale up if in profit
    scale_up_cooldown_seconds: float = 300.0     # 5 min between scale-ups
    scale_down_trigger_loss_eur: float = 20.0    # Scale down if losing this much

    # Trade cooldowns (still used)
    same_direction_cooldown_seconds: float = 300.0   # 5 min between same-direction trades
    reversal_cooldown_seconds: float = 120.0         # 2 min after reversal

    # Signal thresholds (still used)
    min_signal_strength: float = 0.4          # Require stronger signals
    strong_signal_threshold: float = 0.7      # Strong signal for reversals / scale-up
    reversal_signal_threshold: float = 0.65   # Strong signal to exit profitable position


class SmartPositionManager:
    """
    Professional net position management system.

    Responsibilities:
      1. Sync with MT5 to know actual positions.
      2. Prevent duplicate/hedge positions.
      3. Smart entry/exit decisions based on signals + P&L.
      4. Time-aware position management.
      5. Use UnifiedLotCalculator for all lot sizing decisions.
      6. Delegate **all** exit rules to ExitStrategyEngine (single source of truth).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load from YAML if no config provided
        if config is None:
            config = load_config_from_yaml()
        self.config = SmartPositionConfig(**config) if config else SmartPositionConfig()

        # Load unified lot sizing config
        lot_config = load_lot_config_from_yaml()
        if lot_config:
            # Override local config with unified values
            if "max_lot" in lot_config:
                self.config.max_lot_size = float(lot_config["max_lot"])

        self.logger = RotatingLogger(
            "SmartPositionManager",
            log_path="logs/position/smart_manager.log",
            operator_mode=True,
            max_lines=5000,
        )

        # InfoBus for state persistence (used for profit peak persistence)
        try:
            from modules.utils.info_bus import InfoBusManager
            self._smart_bus = InfoBusManager.get_instance()
        except Exception:
            self._smart_bus = None

        # Initialize unified lot calculator (singleton)
        self._lot_calculator = None

        # State tracking
        self._positions: Dict[str, LivePosition] = {}      # Net positions per symbol
        self._actual_mt5_position_count: int = 0           # Actual count of MT5 positions (before netting)
        self._profit_peaks: Dict[str, float] = self._load_peaks_from_bus()  # Track peak profit per symbol (persisted)
        self._last_trade_time: Dict[str, float] = {}       # Per-symbol cooldown
        self._last_scale_time: Dict[str, float] = {}       # Per-symbol scale cooldown
        self._last_sync_time: float = 0

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
    # Position Sync - Core of the system
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
                # Initialize or update profit peak (external view used to seed ExitEngine)
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

        # Persist peaks to InfoBus for restart survival
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

        buy_lots = 0.0
        sell_lots = 0.0
        buy_value = 0.0
        sell_value = 0.0
        total_pnl = 0.0
        earliest_time = float("inf")
        current_price = 0.0
        
        # Track the largest ticket (by lots) for each side for SL/TP modifications
        largest_buy_ticket = 0
        largest_buy_lots = 0.0
        largest_sell_ticket = 0
        largest_sell_lots = 0.0
        
        # Track SL/TP from the largest position
        position_sl = 0.0
        position_tp = 0.0

        for pos in positions:
            lots = float(pos.get("volume", pos.get("lots", 0)) or 0.0)
            entry = float(pos.get("price_open", pos.get("entry_price", 0)) or 0.0)
            pnl = float(pos.get("profit", pos.get("unrealized_pnl", 0)) or 0.0)
            open_time = float(pos.get("time", pos.get("open_time", 0)) or 0.0)
            pos_type = pos.get("type", 0)
            ticket = int(pos.get("ticket", 0) or 0)
            sl = float(pos.get("sl", 0.0) or 0.0)
            tp = float(pos.get("tp", 0.0) or 0.0)

            # Determine side from MT5 type
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
            
            # Track SL/TP from positions with the most lots
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
        
        # Use ticket from the net side's largest position
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
    # Internal helpers for decisions / logging
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

        # For any explicit CLOSE/REVERSE we also reset profit peaks in both
        # the local manager and the unified exit engine, to avoid stale peaks
        # bleeding into future positions on the same symbol.
        if dec.action in (PositionAction.CLOSE, PositionAction.REVERSE):
            if dec.symbol in self._profit_peaks:
                self._profit_peaks.pop(dec.symbol, None)
            try:
                exit_engine = get_exit_engine()
                exit_engine.reset_peak(dec.symbol)
            except Exception:
                pass

        self._log_decision(dec)
        return dec

    def _log_decision(self, decision: SmartDecision) -> None:
        """Log non-HOLD decisions in a compact, operator-friendly format."""
        if decision.action == PositionAction.HOLD:
            return
        try:
            extra_info: Dict[str, Any] = {}
            if decision.new_sl:
                extra_info["new_sl"] = f"{decision.new_sl:.5f}"
            if decision.new_tp:
                extra_info["new_tp"] = f"{decision.new_tp:.5f}"
            if decision.expert_support_ratio != 0.5:
                extra_info["expert_support"] = f"{decision.expert_support_ratio:.0%}"

            self.logger.info(
                format_operator_message(
                    "🎯",
                    "SMART_POSITION_DECISION",
                    symbol=decision.symbol,
                    action=decision.action.value,
                    side=decision.side,
                    lots=f"{decision.lots:.2f}",
                    confidence=f"{decision.confidence:.2f}",
                    reasons=" | ".join(decision.reasons[:3]) if decision.reasons else "",
                    close_first=decision.close_first,
                    **extra_info,
                )
            )
        except Exception:
            # Logging must never break trading logic
            pass

    # =========================================================
    # Smart Decision Engine
    # =========================================================

    def decide(
        self,
        symbol: str,
        signal_direction: int,   # 1 = BUY signal, -1 = SELL signal, 0 = neutral
        signal_strength: float,  # 0.0 to 1.0
        consensus_confidence: float = 0.5,
    ) -> SmartDecision:
        """
        Make a smart position decision.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            signal_direction: 1 for BUY, -1 for SELL, 0 for neutral
            signal_strength: How strong the signal is (0-1)
            consensus_confidence: Voting system confidence (0-1)

        Returns:
            SmartDecision with action, lots, and reasoning
        """
        cfg = self.config

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

        # ─────────────────────────────────────────────────────
        # Case 1: No existing position
        # ─────────────────────────────────────────────────────
        if position is None:
            return self._decide_new_position(
                symbol=symbol,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=consensus_confidence,
                reasons=reasons,
            )

        # Keep trailing P&L peak updated even if sync cadence is imperfect
        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        # ─────────────────────────────────────────────────────
        # Case 2: Have position - check exits using UNIFIED ExitStrategyEngine
        # This ensures consistency with PositionManager (training)
        # ─────────────────────────────────────────────────────

        # Build PositionContext for unified exit evaluation
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
            consensus_confidence=consensus_confidence,
        )

        # Evaluate using unified exit engine (single source of truth)
        exit_engine = get_exit_engine()
        exit_decision = exit_engine.evaluate(exit_ctx)

        if exit_decision.should_exit:
            # Map exit reason to human-readable message
            reason_msg = exit_decision.details.get("message", exit_decision.reason.name)
            reasons.append(f"[ExitEngine] {reason_msg}")

            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,
                confidence=exit_decision.confidence,
                reasons=reasons,
            )

        # ─────────────────────────────────────────────────────
        # Case 3: Position exists, signal aligns - consider scaling
        # ─────────────────────────────────────────────────────
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

        # ─────────────────────────────────────────────────────
        # Case 4: Signal opposes but not handled by exits above
        # ─────────────────────────────────────────────────────
        if signal_against and signal_strength >= cfg.strong_signal_threshold:
            reasons.append(
                f"REVERSAL: Strong opposing signal ({signal_strength:.2f}) "
                f"vs {position.direction} position"
            )

            # If we are losing, we still prioritise closing regardless of cooldown
            if position.unrealized_pnl < 0:
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,  # Pass position side for proper close direction
                    confidence=0.75,
                    reasons=reasons,
                )

            # If we are profitable: obey reversal cooldown
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

            # No cooldown issue: allow REVERSE
            return self._make_decision(
                action=PositionAction.REVERSE,
                symbol=symbol,
                lots=self.config.default_lot_size,
                side=signal_direction,
                confidence=0.70,
                reasons=reasons,
                close_first=True,
            )

        # ─────────────────────────────────────────────────────
        # Case 5: Nothing actionable - HOLD
        # ─────────────────────────────────────────────────────
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
                return self._make_decision(
                    action=PositionAction.HOLD,
                    symbol=symbol,
                    confidence=0.5,
                    reasons=reasons,
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

        # Check signal strength threshold
        if signal_strength < cfg.min_signal_strength:
            reasons.append(
                f"Signal too weak ({signal_strength:.2f} < {cfg.min_signal_strength:.2f})"
            )
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Check neutral signal
        if signal_direction == 0:
            reasons.append("Neutral signal - no direction")
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Check cooldown (same direction)
        last_trade = self._last_trade_time.get(symbol, 0.0)
        cooldown_remaining = (
            cfg.same_direction_cooldown_seconds - (time.time() - last_trade)
        )
        if cooldown_remaining > 0:
            reasons.append(f"Cooldown active ({cooldown_remaining:.0f}s remaining)")
            return self._make_decision(
                action=PositionAction.HOLD,
                symbol=symbol,
                confidence=0.5,
                reasons=reasons,
            )

        # Check total position count - use ACTUAL MT5 count, not net positions
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

        # ═══════════════════════════════════════════════════════════════
        # UNIFIED LOT CALCULATION - Use the central lot calculator
        # ═══════════════════════════════════════════════════════════════
        lots = 0.0
        if self.lot_calculator and self.config.use_unified_calculator:
            try:
                lots, lot_details = self.lot_calculator.calculate_lots(
                    symbol=symbol,
                    signal_strength=signal_strength,
                )
                self.logger.info(
                    f"[SMART_PM] 📊 UNIFIED_LOT: {symbol} | signal={signal_strength:.2f} | "
                    f"lots={lots:.2f} | balance=€{lot_details.get('balance', 0):.0f} | "
                    f"risk={lot_details.get('risk_pct', 0)*100:.1f}%"
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

        reasons.append(
            f"OPEN {action.value}: signal={signal_strength:.2f}, "
            f"consensus={consensus_confidence:.2f}, lots={lots:.2f}"
        )

        return self._make_decision(
            action=action,
            symbol=symbol,
            lots=round(lots, 2),
            side=signal_direction,
            confidence=min(signal_strength, consensus_confidence),
            reasons=reasons,
        )

    def _decide_scale(
        self,
        symbol: str,
        position: LivePosition,
        signal_strength: float,
        consensus_confidence: float,
        reasons: List[str],
    ) -> SmartDecision:
        """Decide on scaling an existing position."""
        cfg = self.config

        # Scale UP conditions: in profit, strong signal, not recently scaled
        last_scale = self._last_scale_time.get(symbol, 0.0)
        scale_cooldown_ok = (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds

        if (
            position.unrealized_pnl >= cfg.scale_up_min_profit_eur
            and signal_strength >= cfg.strong_signal_threshold
            and scale_cooldown_ok
            and position.lots < cfg.max_lot_size
        ):
            # ═══════════════════════════════════════════════════════════════
            # UNIFIED LOT CALCULATION for scale-up
            # ═══════════════════════════════════════════════════════════════
            add_lots = 0.0
            if self.lot_calculator and self.config.use_unified_calculator:
                try:
                    # Use lower signal strength for scale-ups (more conservative)
                    scale_signal = signal_strength * 0.5
                    add_lots, _ = self.lot_calculator.calculate_lots(
                        symbol=symbol,
                        signal_strength=scale_signal,
                    )
                    # Cap by remaining capacity
                    add_lots = min(add_lots, cfg.max_lot_size - position.lots)
                except Exception:
                    add_lots = 0.0

            # Fallback
            if add_lots <= 0:
                add_lots = min(
                    cfg.default_lot_size * 0.5,  # Conservative add
                    cfg.max_lot_size - position.lots,
                )

            if add_lots >= 0.01:  # Lowered from 0.1 to allow smaller scales
                reasons.append(
                    f"SCALE UP: +€{position.unrealized_pnl:.2f} profit, "
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

        # Scale DOWN conditions: losing with aligning signal (reduce exposure)
        if (
            position.unrealized_pnl <= -cfg.scale_down_trigger_loss_eur
            and position.lots > 0.01
        ):
            reduce_lots = min(position.lots * 0.5, max(position.lots - 0.01, 0.0))

            if reduce_lots >= 0.01:
                reasons.append(
                    f"SCALE DOWN: -€{abs(position.unrealized_pnl):.2f} loss, "
                    f"reducing exposure"
                )
                return self._make_decision(
                    action=PositionAction.SCALE_DOWN,
                    symbol=symbol,
                    lots=round(reduce_lots, 2),
                    side=position.side,
                    confidence=0.65,
                    reasons=reasons,
                )

        # No scaling action - hold
        reasons.append(
            f"HOLD (aligning): {position.direction} {position.lots:.2f} lots, "
            f"P&L €{position.unrealized_pnl:.2f}"
        )
        return self._make_decision(
            action=PositionAction.HOLD,
            symbol=symbol,
            confidence=0.60,
            reasons=reasons,
        )

    # =========================================================
    # Position-Focused Management (COOPERATIVE MODE)
    # =========================================================
    # When a position is open, the ENTIRE system focuses on managing it:
    # - All experts evaluate FOR this position (support/oppose)
    # - Exit decisions always flow through ExitStrategyEngine
    # - Experts modulate non-critical exits (tighten vs close)
    # =========================================================

    def manage_position(
        self,
        signal: PositionManagementSignal,
    ) -> SmartDecision:
        """
        Position-focused decision making with full expert context.

        This is the COOPERATIVE MODE - when a position exists, the entire
        pipeline (experts, voting, PPO) works to protect and manage it.

        Exit rules themselves are **always** delegated to ExitStrategyEngine.
        Experts only modulate non-critical exits and scaling / SL adjustments.
        """
        symbol = signal.symbol
        position = self._positions.get(symbol)
        cfg = self.config
        reasons: List[str] = []

        # No position = delegate to regular decide()
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

        # Update local peak (used to seed ExitEngine peak logic)
        current_peak = self._profit_peaks.get(symbol, position.unrealized_pnl)
        if position.unrealized_pnl > current_peak:
            self._profit_peaks[symbol] = position.unrealized_pnl

        # Determine expert alignment with position
        position_action = "BUY" if position.side > 0 else "SELL"
        supporting_experts: List[ExpertSignal] = []
        opposing_experts: List[ExpertSignal] = []

        for expert in signal.expert_signals:
            # For position-focused mode, "supporting" means aligned with the
            # current side or explicitly HOLD; everything else is opposing.
            if expert.action == position_action or expert.action == "HOLD":
                supporting_experts.append(expert)
            else:
                opposing_experts.append(expert)

        total_experts = len(signal.expert_signals)
        support_ratio = (
            len(supporting_experts) / total_experts if total_experts > 0 else 0.5
        )

        # Build management context (enriched with ExitEngine outputs later)
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

        # ─────────────────────────────────────────────────────
        # PRIORITY 0: Unified ExitEngine decision
        # ─────────────────────────────────────────────────────

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
            signal_strength=float(
                max(0.0, min(1.0, signal.consensus_confidence))
            ),
            consensus_confidence=float(
                max(0.0, min(1.0, signal.consensus_score))
            ),
        )

        exit_engine = get_exit_engine()
        exit_decision = exit_engine.evaluate(exit_ctx)

        management_ctx["exit_engine_reason"] = exit_decision.reason.name
        management_ctx["exit_engine_should_exit"] = exit_decision.should_exit
        management_ctx["exit_engine_confidence"] = exit_decision.confidence

        if exit_decision.should_exit:
            engine_msg = exit_decision.details.get(
                "message", exit_decision.reason.name
            )
            reasons.append(f"[ExitEngine] {engine_msg}")

            # Critical exits (HARD_STOP / EMERGENCY / TIME_DECAY) are **never**
            # negotiated with experts. They always close.
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

            # Non-critical exits: let experts influence whether we
            # honour the close strictly or tighten protection instead.
            if support_ratio <= 0.3 or position.unrealized_pnl <= 0:
                # Most experts against the position OR we are not in profit:
                # follow ExitEngine and close.
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

            # Experts mostly support and position is profitable:
            # prefer tightening protection instead of immediate exit.
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

            # Fallback: if we cannot compute a better SL, respect ExitEngine
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

        # At this point ExitEngine wants HOLD: we are in pure
        # cooperative-management mode (scale / SL only).

        # ─────────────────────────────────────────────────────
        # PRIORITY 1: Strong expert support + profitable trade
        # ─────────────────────────────────────────────────────

        if support_ratio >= 0.6 and position.unrealized_pnl > 0:
            # 1a) Scale up if very strong support AND good profit
            if (
                support_ratio >= 0.75
                and position.unrealized_pnl >= cfg.scale_up_min_profit_eur
            ):
                last_scale = self._last_scale_time.get(symbol, 0.0)
                if (time.time() - last_scale) >= cfg.scale_up_cooldown_seconds:
                    if position.lots < cfg.max_lot_size:
                        add_lots = self._calculate_scale_lots(position, signal)
                        if add_lots >= 0.01:
                            reasons.append(
                                f"📈 SCALE UP: {int(support_ratio * 100)}% expert support, "
                                f"+€{position.unrealized_pnl:.2f} profit"
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

            # 1b) Lock profit by moving SL to breakeven+
            if position.unrealized_pnl >= 30:  # €30+ profit
                new_sl = self._calculate_breakeven_sl(position)
                if new_sl is not None and self._is_sl_improvement(position, new_sl):
                    reasons.append(
                        f"🔒 LOCK PROFIT: Moving SL to breakeven+ "
                        f"(€{position.unrealized_pnl:.2f} profit, "
                        f"{int(support_ratio * 100)}% expert support)"
                    )
                    return self._make_decision(
                        action=PositionAction.ADJUST_SL,
                        symbol=symbol,
                        side=position.side,
                        confidence=0.65,
                        reasons=reasons,
                        new_sl=new_sl,
                        management_context=management_ctx,
                        expert_support_ratio=support_ratio,
                    )

        # ─────────────────────────────────────────────────────
        # PRIORITY 2: Mixed expert signals + loss
        # ─────────────────────────────────────────────────────

        if 0.3 <= support_ratio < 0.6:
            if position.unrealized_pnl <= -cfg.scale_down_trigger_loss_eur:
                reasons.append(
                    f"⚠️ MIXED SIGNALS + LOSS: Only {int(support_ratio * 100)}% support, "
                    f"loss €{position.unrealized_pnl:.2f}"
                )

                if signal.fragility > 0.7:
                    # High fragility = unstable consensus => prefer clean exit
                    return self._make_decision(
                        action=PositionAction.CLOSE,
                        symbol=symbol,
                        side=position.side,
                        confidence=0.70,
                        reasons=reasons,
                        management_context=management_ctx,
                        expert_support_ratio=support_ratio,
                    )
                else:
                    # Scale down instead of full close
                    reduce_lots = min(position.lots * 0.5, position.lots - 0.01)
                    if reduce_lots >= 0.01:
                        return self._make_decision(
                            action=PositionAction.SCALE_DOWN,
                            symbol=symbol,
                            lots=round(reduce_lots, 2),
                            side=position.side,
                            confidence=0.65,
                            reasons=reasons,
                            management_context=management_ctx,
                            expert_support_ratio=support_ratio,
                        )

        # ─────────────────────────────────────────────────────
        # DEFAULT: Continue holding with current settings
        # ─────────────────────────────────────────────────────

        reasons.append(
            f"✓ HOLD: {position.direction} {position.lots:.2f} lots | "
            f"PnL €{position.unrealized_pnl:.2f} | "
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
    # Helper methods for SL / scaling in management mode
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

            # For a LONG: SL should be below current price
            # For a SHORT: SL should be above current price
            # Tighten to protect ~50% of current profit
            profit_per_unit = position.current_price - position.entry_price
            if position.side < 0:  # SHORT
                profit_per_unit = position.entry_price - position.current_price

            # Protect half the profit
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

            # Add small buffer above breakeven
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
                return True  # No current SL, any SL is improvement

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

            # Base scale amount
            if self.lot_calculator and cfg.use_unified_calculator:
                scale_signal = (
                    signal.consensus_confidence * signal.support_ratio * 0.5
                )
                add_lots, _ = self.lot_calculator.calculate_lots(
                    symbol=position.symbol,
                    signal_strength=scale_signal,
                )
            else:
                add_lots = cfg.default_lot_size * signal.support_ratio * 0.5

            # Cap by remaining capacity
            max_add = cfg.max_lot_size - position.lots
            add_lots = min(add_lots, max_add, cfg.default_lot_size)

            return round(max(0.01, add_lots), 2)
        except Exception:
            return 0.0

    # =========================================================
    # Utility Methods
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

        # Group by symbol
        by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for pos in mt5_positions:
            symbol = pos.get("symbol", "")
            if symbol:
                by_symbol.setdefault(symbol, []).append(pos)

        for symbol, positions in by_symbol.items():
            if len(positions) <= 1:
                continue

            # Check for hedging (both buy and sell)
            buys = [p for p in positions if p.get("type", 0) == 0]
            sells = [p for p in positions if p.get("type", 0) == 1]

            if buys and sells:
                # We have hedging - close the smaller side
                buy_lots = sum(float(p.get("volume", 0) or 0.0) for p in buys)
                sell_lots = sum(float(p.get("volume", 0) or 0.0) for p in sells)

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

        Important for detecting hedge situations where we have many
        offsetting positions that net to zero.
        """
        return self._actual_mt5_position_count

    def has_hedged_positions(self) -> bool:
        """
        Check if we have hedged positions (actual count > net count).

        Returns True if there are more actual MT5 positions than net positions,
        indicating hedging is occurring.
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
