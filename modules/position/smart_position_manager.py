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
# -------------------------------------------------------------

from __future__ import annotations

import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

from modules.utils.audit_utils import RotatingLogger, format_operator_message


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


class PositionAction(Enum):
    """Clean action types for position management."""
    HOLD = "HOLD"
    OPEN_LONG = "OPEN_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE = "CLOSE"
    SCALE_UP = "SCALE_UP"
    SCALE_DOWN = "SCALE_DOWN"
    REVERSE = "REVERSE"  # Close + Open opposite


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "symbol": self.symbol,
            "lots": self.lots,
            "side": self.side,
            "confidence": self.confidence,
            "reasons": self.reasons,
            "close_first": self.close_first,
        }


@dataclass
class SmartPositionConfig:
    """Configuration for smart position management."""
    # Position limits
    max_positions_per_symbol: int = 1
    max_total_positions: int = 4
    default_lot_size: float = 0.2
    max_lot_size: float = 1.0

    # Profit-taking thresholds
    profit_take_activation_eur: float = 50.0  # Start trailing after €50 profit
    profit_take_trail_pct: float = 0.35      # Close if drops 35% from peak
    momentum_exit_profit_eur: float = 40.0   # Take profit on signal reversal above €40

    # Loss-cutting thresholds
    hard_stop_loss_eur: float = 100.0  # Absolute max loss
    soft_stop_loss_eur: float = 30.0   # Cut early if signal against
    time_decay_stop_eur: float = 50.0  # Cut if losing AND old
    time_decay_hours: float = 4.0      # Position age for time decay stop

    # Scaling rules
    scale_up_min_profit_eur: float = 10.0        # Only scale up if in profit
    scale_up_cooldown_seconds: float = 300.0     # 5 min between scale-ups
    scale_down_trigger_loss_eur: float = 20.0    # Scale down if losing this much

    # Trade cooldowns
    same_direction_cooldown_seconds: float = 300.0   # 5 min between same-direction trades
    reversal_cooldown_seconds: float = 120.0         # 2 min after reversal

    # Signal thresholds
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
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load from YAML if no config provided
        if config is None:
            config = load_config_from_yaml()
        self.config = SmartPositionConfig(**config) if config else SmartPositionConfig()
        self.logger = RotatingLogger(
            "SmartPositionManager",
            log_path="logs/position/smart_manager.log",
            operator_mode=True,
            max_lines=5000,
        )

        # State tracking
        self._positions: Dict[str, LivePosition] = {}      # Net positions per symbol
        self._actual_mt5_position_count: int = 0           # Actual count of MT5 positions (before netting)
        self._profit_peaks: Dict[str, float] = {}          # Track peak profit per symbol
        self._last_trade_time: Dict[str, float] = {}       # Per-symbol cooldown
        self._last_scale_time: Dict[str, float] = {}       # Per-symbol scale cooldown
        self._last_sync_time: float = 0

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
                # Initialize or update profit peak
                prev_peak = self._profit_peaks.get(symbol, net_pos.unrealized_pnl)
                self._profit_peaks[symbol] = max(prev_peak, net_pos.unrealized_pnl)

        # Clean up profit peaks for closed positions
        closed_symbols = set(self._profit_peaks.keys()) - set(self._positions.keys())
        for symbol in closed_symbols:
            del self._profit_peaks[symbol]

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

        for pos in positions:
            lots = float(pos.get("volume", pos.get("lots", 0)) or 0.0)
            entry = float(pos.get("price_open", pos.get("entry_price", 0)) or 0.0)
            pnl = float(pos.get("profit", pos.get("unrealized_pnl", 0)) or 0.0)
            open_time = float(pos.get("time", pos.get("open_time", 0)) or 0.0)
            pos_type = pos.get("type", 0)

            # Determine side from MT5 type
            is_buy = pos_type == 0  # POSITION_TYPE_BUY = 0

            if is_buy:
                buy_lots += lots
                buy_value += lots * entry
            else:
                sell_lots += lots
                sell_value += lots * entry

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

        # VWAP entry price for net side
        if side > 0 and buy_lots > 0:
            entry_price = buy_value / buy_lots
        elif side < 0 and sell_lots > 0:
            entry_price = sell_value / sell_lots
        else:
            entry_price = current_price

        return LivePosition(
            symbol=symbol,
            side=side,
            lots=abs_lots,
            entry_price=entry_price,
            current_price=current_price,
            unrealized_pnl=total_pnl,
            open_time=earliest_time if earliest_time < float("inf") else time.time(),
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
    ) -> SmartDecision:
        dec = SmartDecision(
            action=action,
            symbol=symbol,
            lots=lots,
            side=side,
            confidence=confidence,
            reasons=reasons or [],
            close_first=close_first,
        )
        self._log_decision(dec)
        return dec

    def _log_decision(self, decision: SmartDecision) -> None:
        """Log non-HOLD decisions in a compact, operator-friendly format."""
        if decision.action == PositionAction.HOLD:
            return
        try:
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
        # Case 2: Have position - check exits first (priority)
        # ─────────────────────────────────────────────────────

        # 2a. Hard stop loss - ALWAYS exit
        if position.unrealized_pnl <= -cfg.hard_stop_loss_eur:
            reasons.append(
                f"HARD STOP: Loss €{position.unrealized_pnl:.2f} "
                f"exceeds -€{cfg.hard_stop_loss_eur:.0f}"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,  # Pass position side for proper close direction
                confidence=0.95,
                reasons=reasons,
            )

        # 2b. Soft stop - exit if losing AND signal against us
        signal_against = (position.side > 0 and signal_direction < 0) or (
            position.side < 0 and signal_direction > 0
        )

        if (
            position.unrealized_pnl <= -cfg.soft_stop_loss_eur
            and signal_against
        ):
            reasons.append(
                f"SOFT STOP: Loss €{position.unrealized_pnl:.2f} with opposing signal "
                f"(strength={signal_strength:.2f})"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,  # Pass position side for proper close direction
                confidence=0.85,
                reasons=reasons,
            )

        # 2c. Time decay stop - old losing position
        if (
            position.age_hours >= cfg.time_decay_hours
            and position.unrealized_pnl <= -cfg.time_decay_stop_eur
        ):
            reasons.append(
                f"TIME DECAY: Position {position.age_hours:.1f}h old, "
                f"loss €{position.unrealized_pnl:.2f}"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,  # Pass position side for proper close direction
                confidence=0.80,
                reasons=reasons,
            )

        # 2d. Trailing profit - take profit on retrace
        peak_profit = self._profit_peaks.get(symbol, 0.0)
        if peak_profit >= cfg.profit_take_activation_eur:
            retrace = (peak_profit - position.unrealized_pnl) / peak_profit
            if retrace >= cfg.profit_take_trail_pct:
                reasons.append(
                    f"TRAILING TP: Peak €{peak_profit:.2f} -> €{position.unrealized_pnl:.2f} "
                    f"({retrace:.1%} retrace)"
                )
                return self._make_decision(
                    action=PositionAction.CLOSE,
                    symbol=symbol,
                    side=position.side,  # Pass position side for proper close direction
                    confidence=0.85,
                    reasons=reasons,
                )

        # 2e. Momentum exit - take profit when signal reverses
        if (
            position.is_profitable
            and signal_against
            and signal_strength >= cfg.reversal_signal_threshold
            and position.unrealized_pnl >= cfg.momentum_exit_profit_eur
        ):
            reasons.append(
                f"MOMENTUM EXIT: +€{position.unrealized_pnl:.2f} with reversal signal "
                f"(strength={signal_strength:.2f})"
            )
            return self._make_decision(
                action=PositionAction.CLOSE,
                symbol=symbol,
                side=position.side,  # Pass position side for proper close direction
                confidence=0.80,
                reasons=reasons,
            )

        # ─────────────────────────────────────────────────────
        # Case 3: Position exists, signal aligns - consider scaling
        # ─────────────────────────────────────────────────────
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
        cooldown_remaining = cfg.same_direction_cooldown_seconds - (time.time() - last_trade)
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
        action = PositionAction.OPEN_LONG if signal_direction > 0 else PositionAction.OPEN_SHORT
        raw_lots = cfg.default_lot_size * signal_strength
        lots = min(max(raw_lots, 0.01), cfg.max_lot_size)  # enforce broker min lot ~0.01

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
            add_lots = min(
                cfg.default_lot_size * 0.5,  # Conservative add
                cfg.max_lot_size - position.lots,
            )

            if add_lots >= 0.1:
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
            and position.lots > cfg.default_lot_size
        ):
            reduce_lots = min(position.lots * 0.5, position.lots - cfg.default_lot_size)

            if reduce_lots >= 0.1:
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
            confidence=0.6,
            reasons=reasons,
        )

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

        return any(count > self.config.max_positions_per_symbol for count in by_symbol.values())

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
