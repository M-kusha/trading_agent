# -------------------------------------------------------------
# File: modules/position/exit_engine.py
# ExitStrategyEngine — Unified exit logic for BOTH systems
#
# This is the SINGLE SOURCE OF TRUTH for position exit decisions.
# Used by:
#   - PositionManager (training/simulation)
#   - SmartPositionManager (live trading via Executor)
#
# Exit Strategies (Priority Order):
#   0. EMERGENCY: Account-level protection (drawdown / daily loss / open risk)
#   1. HARD_STOP: Absolute max loss (€150 default) - ALWAYS EXIT
#   2. SOFT_STOP: Loss + opposing signal - EXIT EARLY
#   3. TIME_DECAY: Old position + losing OR stale profit
#   4. TRAILING_PROFIT: Dynamic ATR/€-based trailing TP
#   5. MOMENTUM_EXIT: Signal reversal while profitable
#   6. SIGNAL_EXIT: Agent direction flip or very weak signal
#
# Regime-aware & confidence-aware:
#   - Thresholds are scaled by market regime (volatile / ranging / trending)
#   - Non-critical exits modulate confidence/urgency with consensus_confidence
# -------------------------------------------------------------

from __future__ import annotations

import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
from enum import Enum, auto


class ExitReason(Enum):
    """Exit strategy that triggered the close decision."""
    HOLD = auto()              # No exit triggered
    HARD_STOP = auto()         # Absolute max loss
    SOFT_STOP = auto()         # Loss + opposing signal
    TIME_DECAY = auto()        # Old position + losing / stale profit
    TRAILING_PROFIT = auto()   # Retraced from profit peak
    MOMENTUM_EXIT = auto()     # Signal reversal while profitable
    SIGNAL_EXIT = auto()       # Agent direction flip / weak signal
    EMERGENCY = auto()         # Emergency conditions (drawdown/exposure)
    MANUAL = auto()            # Explicit close request


@dataclass
class ExitDecision:
    """Result of exit strategy evaluation."""
    should_exit: bool
    reason: ExitReason
    confidence: float  # 0.0 to 1.0
    urgency: float     # 0.0 (can wait) to 1.0 (immediate)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        """Critical exits should never be ignored."""
        return self.reason in (
            ExitReason.HARD_STOP,
            ExitReason.EMERGENCY,
            ExitReason.TIME_DECAY,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_exit": self.should_exit,
            "reason": self.reason.name,
            "confidence": self.confidence,
            "urgency": self.urgency,
            "is_critical": self.is_critical,
            "details": self.details,
        }


@dataclass
class ExitConfig:
    """
    Exit strategy configuration.

    Loaded from risk_policy.yaml -> exit_strategies section.
    Falls back to smart_position section for backward compatibility.
    """
    # Hard stop - ALWAYS EXIT (prop firm protection)
    hard_stop_loss_eur: float = 150.0

    # Soft stop - exit if losing AND signal against
    soft_stop_loss_eur: float = 80.0
    soft_stop_min_signal: float = 0.3  # Minimum opposing signal strength

    # Time decay - old losing positions are drag
    time_decay_hours: float = 4.0
    time_decay_stop_eur: float = 60.0

    # Trailing profit - ATR/€-based dynamic trailing
    trailing_activation_eur: float = 100.0   # Start trailing after €100 profit
    trailing_activation_atr: float = 2.0     # Or after 2x ATR (in €)
    trailing_retrace_pct: float = 0.30       # Close if retraces 30% from peak
    trailing_retrace_atr: float = 1.5        # Or if retraces 1.5x ATR (in €)
    trailing_use_atr: bool = True            # Prefer ATR-based (smarter)
    trailing_min_peak_eur: float = 50.0      # Don't trail if peak was tiny

    # Momentum exit - take profit on signal reversal
    momentum_exit_profit_eur: float = 60.0
    momentum_reversal_signal: float = 0.65   # Signal strength to trigger

    # Signal exit - agent wants out
    signal_exit_threshold: float = 0.10      # Very weak signal = exit
    signal_direction_weight: float = 0.8     # How much to trust direction flip

    # Regime-adaptive scaling
    volatile_regime_tighten: float = 0.7     # Tighten stops 30% in volatile
    ranging_regime_loosen: float = 1.2       # Loosen stops 20% in ranging
    trending_regime_neutral: float = 1.0     # Normal stops in trending/normal

    # Emergency / account protection
    emergency_drawdown_pct: float = 0.08          # 8% account DD triggers exit
    emergency_daily_loss_buffer_pct: float = 0.9  # 90% of daily loss limit
    emergency_max_open_risk_eur: float = 2000.0   # Failsafe for total open risk


def load_exit_config() -> ExitConfig:
    """Load exit configuration from risk_policy.yaml."""
    config_path = Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            # Try new exit_strategies section first
            exit_cfg = policy.get("exit_strategies", {})

            # Fall back to smart_position for backward compatibility
            if not exit_cfg:
                smart = policy.get("smart_position", {})
                exit_cfg = {
                    "hard_stop_loss_eur": smart.get("hard_stop_loss_eur", 150.0),
                    "soft_stop_loss_eur": smart.get("soft_stop_loss_eur", 80.0),
                    "time_decay_hours": smart.get("time_decay_hours", 4.0),
                    "time_decay_stop_eur": smart.get("time_decay_stop_eur", 60.0),
                    "trailing_activation_eur": smart.get(
                        "profit_take_activation_eur", 100.0
                    ),
                    "trailing_retrace_pct": smart.get(
                        "profit_take_trail_pct", 0.30
                    ),
                    "momentum_exit_profit_eur": smart.get(
                        "momentum_exit_profit_eur", 60.0
                    ),
                    "momentum_reversal_signal": smart.get(
                        "reversal_signal_threshold", 0.65
                    ),
                }

            # Only pass keys that exist on ExitConfig
            return ExitConfig(**{
                k: v for k, v in exit_cfg.items()
                if hasattr(ExitConfig, k)
            })
    except Exception as e:
        print(f"[ExitEngine] Failed to load config: {e}")

    return ExitConfig()


@dataclass
class PositionContext:
    """Context about the current position for exit evaluation."""
    # Position-level
    symbol: str
    side: int                      # +1 = long, -1 = short
    unrealized_pnl: float          # Current P&L in EUR
    peak_pnl: float                # Highest P&L reached (per position)
    entry_price: float
    current_price: float
    open_time: float               # Unix timestamp
    lots: float = 0.0
    position_id: str = ""          # Optional unique id for this position/ticket

    # Market context
    atr: Optional[float] = None    # Current ATR (in price units)
    volatility: float = 0.02       # Normalized volatility (e.g. stdev of returns)
    regime: str = "normal"         # "volatile", "ranging", "trending", "normal", "auto"

    # Signal / committee context
    signal_direction: int = 0      # +1 = bullish, -1 = bearish, 0 = neutral
    signal_strength: float = 0.0   # 0.0 to 1.0
    signal_valid: bool = False     # True if signal has been calculated (not default/startup)
    consensus_confidence: float = 0.5  # 0.0 (no trust) to 1.0 (strong consensus)

    # Account / portfolio context (optional, for EMERGENCY exits)
    account_drawdown_pct: Optional[float] = None   # 0.05 = 5% from equity peak
    daily_loss_eur: Optional[float] = None         # Signed P&L today (negative = loss)
    daily_loss_limit_eur: Optional[float] = None   # Allowed daily loss (>0)
    total_open_risk_eur: Optional[float] = None    # Sum of open risk across all trades

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
    def signal_against(self) -> bool:
        """True if signal direction opposes position side."""
        if self.signal_direction == 0:
            return False
        return ((self.side > 0 and self.signal_direction < 0) or
                (self.side < 0 and self.signal_direction > 0))

    @property
    def signal_aligns(self) -> bool:
        """True if signal direction aligns with position side."""
        if self.signal_direction == 0:
            return False
        return ((self.side > 0 and self.signal_direction > 0) or
                (self.side < 0 and self.signal_direction < 0))

    @property
    def pnl_pips(self) -> float:
        """Approximate P&L in pips (for logging only)."""
        if self.entry_price == 0:
            return 0.0

        symbol = self.symbol.upper()
        if "XAU" in symbol:
            # Treat $0.1 move as "1 pip" equivalent for logging
            pip_size = 0.1
        elif "JPY" in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001

        return (self.current_price - self.entry_price) * self.side / pip_size


class ExitStrategyEngine:
    """
    Unified exit strategy engine.

    Evaluates all exit strategies and returns the highest-priority
    exit decision. This is the SINGLE SOURCE OF TRUTH for exit logic.
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        self.config = config or load_exit_config()
        # Track peak P&L per position; key = "position_id|SYMBOL" or "SYMBOL"
        self._profit_peaks: Dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _peak_key(self, ctx: PositionContext) -> str:
        """Internal key for peak tracking."""
        if ctx.position_id:
            return f"{ctx.position_id}|{ctx.symbol}"
        return ctx.symbol

    def _update_peak(self, ctx: PositionContext) -> None:
        """Track profit peak for trailing stop."""
        key = self._peak_key(ctx)
        current_peak = self._profit_peaks.get(key, ctx.unrealized_pnl)
        self._profit_peaks[key] = max(current_peak, ctx.unrealized_pnl)

    def _effective_peak_pnl(self, ctx: PositionContext) -> float:
        """
        Compute effective peak P&L with stale-peak protection.

        Combines externally supplied ctx.peak_pnl (e.g. persisted by
        SmartPositionManager) with internally tracked peaks and resets
        obviously stale values.
        """
        self._update_peak(ctx)
        peak_key = self._peak_key(ctx)
        tracked_peak = self._profit_peaks.get(peak_key, ctx.unrealized_pnl)

        # If the position is very new and near flat P&L, but the stored peak
        # is very large, treat the stored value as stale (leftover from a
        # previous position) and reset it.
        if (
            ctx.age_seconds < 300
            and abs(ctx.unrealized_pnl) < 50.0
            and tracked_peak > 100.0
        ):
            self._profit_peaks[peak_key] = ctx.unrealized_pnl
            tracked_peak = ctx.unrealized_pnl

        return max(ctx.peak_pnl, tracked_peak)

    def _adjust_confidence(self, base_conf: float, ctx: PositionContext) -> float:
        """
        Modulate confidence with committee consensus.

        factor = 0.5 + 0.5 * consensus_confidence
        """
        factor = 0.5 + 0.5 * max(0.0, min(ctx.consensus_confidence, 1.0))
        conf = base_conf * factor
        return max(0.0, min(conf, 1.0))

    def _adjust_urgency(self, base_urg: float, ctx: PositionContext) -> float:
        """
        Modulate urgency with volatility.

        Higher volatility => slightly higher urgency.
        Very low volatility => slightly lower urgency.
        """
        vol = ctx.volatility
        if vol >= 0.03:
            factor = 1.1
        elif vol <= 0.01:
            factor = 0.9
        else:
            factor = 1.0

        urg = base_urg * factor
        return max(0.0, min(urg, 1.0))

    def _regime_adjusted_config(self, ctx: PositionContext) -> ExitConfig:
        """
        Adjust config thresholds based on market regime and volatility.

        - If ctx.regime is explicitly set (volatile/ranging/trending), use it.
        - If ctx.regime is "auto"/"normal", infer from ctx.volatility.
        """
        regime = (ctx.regime or "normal").lower()

        if regime in ("auto", "normal", ""):
            # Simple volatility-based routing if regime not explicitly set
            vol = ctx.volatility
            if vol >= 0.03:
                regime = "volatile"
            elif vol <= 0.01:
                regime = "ranging"
            else:
                regime = "trending"

        # Get scaling factor
        if regime in ("volatile", "high_volatility"):
            scale = self.config.volatile_regime_tighten      # <1.0 => tighter
        elif regime in ("ranging", "sideways", "low_volatility"):
            scale = self.config.ranging_regime_loosen        # >1.0 => looser
        else:
            scale = self.config.trending_regime_neutral      # 1.0

        if scale == 1.0:
            return self.config

        # Create adjusted config; numeric thresholds scaled appropriately
        return ExitConfig(
            hard_stop_loss_eur=self.config.hard_stop_loss_eur * scale,
            soft_stop_loss_eur=self.config.soft_stop_loss_eur * scale,
            soft_stop_min_signal=self.config.soft_stop_min_signal,
            time_decay_hours=self.config.time_decay_hours,  # Do not scale time
            time_decay_stop_eur=self.config.time_decay_stop_eur * scale,
            trailing_activation_eur=self.config.trailing_activation_eur,
            trailing_activation_atr=self.config.trailing_activation_atr,
            trailing_retrace_pct=self.config.trailing_retrace_pct * scale,
            trailing_retrace_atr=self.config.trailing_retrace_atr * scale,
            trailing_use_atr=self.config.trailing_use_atr,
            trailing_min_peak_eur=self.config.trailing_min_peak_eur * scale,
            momentum_exit_profit_eur=self.config.momentum_exit_profit_eur * scale,
            momentum_reversal_signal=self.config.momentum_reversal_signal,
            signal_exit_threshold=self.config.signal_exit_threshold,
            signal_direction_weight=self.config.signal_direction_weight,
            volatile_regime_tighten=self.config.volatile_regime_tighten,
            ranging_regime_loosen=self.config.ranging_regime_loosen,
            trending_regime_neutral=self.config.trending_regime_neutral,
            emergency_drawdown_pct=self.config.emergency_drawdown_pct,
            emergency_daily_loss_buffer_pct=self.config.emergency_daily_loss_buffer_pct,
            emergency_max_open_risk_eur=self.config.emergency_max_open_risk_eur,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def evaluate(self, ctx: PositionContext) -> ExitDecision:
        """
        Evaluate all exit strategies for a position.

        Returns the highest-priority exit decision.
        Strategies are evaluated in priority order.
        """
        # Effective peak P&L (combining external and internal tracking)
        ctx_peak = self._effective_peak_pnl(ctx)

        # Regime-adjusted config
        cfg = self._regime_adjusted_config(ctx)

        # 0. EMERGENCY - account-level protection
        emergency_decision = self._check_emergency(ctx, cfg)
        if emergency_decision.should_exit:
            return emergency_decision

        # 1. HARD STOP - Always check early (prop firm protection)
        if ctx.unrealized_pnl <= -cfg.hard_stop_loss_eur:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.HARD_STOP,
                confidence=0.99,
                urgency=1.0,
                details={
                    "loss": ctx.unrealized_pnl,
                    "threshold": -cfg.hard_stop_loss_eur,
                    "message": (
                        f"HARD STOP: Loss €{ctx.unrealized_pnl:.2f} "
                        f"exceeds -€{cfg.hard_stop_loss_eur:.0f}"
                    ),
                },
            )

        # 2. SOFT STOP - Loss + opposing signal (shared implementation)
        soft_stop_decision = self._check_soft_stop(ctx, cfg)
        if soft_stop_decision.should_exit:
            return soft_stop_decision

        # 3. TIME DECAY - Old position + losing OR stale profit
        time_decay_decision = self._check_time_decay(ctx, ctx_peak, cfg)
        if time_decay_decision.should_exit:
            return time_decay_decision

        # 4. TRAILING PROFIT - Dynamic trailing from peak
        trailing_decision = self._check_trailing_profit(ctx, ctx_peak, cfg)
        if trailing_decision.should_exit:
            return trailing_decision

        # 5. MOMENTUM EXIT - Take profit on signal reversal
        momentum_decision = self._check_momentum_exit(ctx, cfg)
        if momentum_decision.should_exit:
            return momentum_decision

        # 6. SIGNAL EXIT - Agent direction flip or very weak signal
        signal_exit = self._check_signal_exit(ctx, cfg)
        if signal_exit.should_exit:
            return signal_exit

        # No exit triggered - HOLD
        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.60,
            urgency=0.0,
            details={
                "pnl": ctx.unrealized_pnl,
                "age_hours": ctx.age_hours,
                "signal_direction": ctx.signal_direction,
                "peak_pnl": ctx_peak,
            },
        )

    def evaluate_all(self, ctx: PositionContext) -> Dict[str, ExitDecision]:
        """
        Diagnostics helper: evaluate all strategies and return their decisions.

        Keys:
            - "emergency", "hard_stop", "soft_stop",
              "time_decay", "trailing", "momentum", "signal", "final"
        """
        ctx_peak = self._effective_peak_pnl(ctx)
        cfg = self._regime_adjusted_config(ctx)

        results: Dict[str, ExitDecision] = {}

        results["emergency"] = self._check_emergency(ctx, cfg)

        # Hard stop
        if ctx.unrealized_pnl <= -cfg.hard_stop_loss_eur:
            results["hard_stop"] = ExitDecision(
                should_exit=True,
                reason=ExitReason.HARD_STOP,
                confidence=0.99,
                urgency=1.0,
                details={
                    "loss": ctx.unrealized_pnl,
                    "threshold": -cfg.hard_stop_loss_eur,
                },
            )
        else:
            results["hard_stop"] = ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                confidence=0.0,
                urgency=0.0,
                details={},
            )

        results["soft_stop"] = self._check_soft_stop(ctx, cfg)
        results["time_decay"] = self._check_time_decay(ctx, ctx_peak, cfg)
        results["trailing"] = self._check_trailing_profit(ctx, ctx_peak, cfg)
        results["momentum"] = self._check_momentum_exit(ctx, cfg)
        results["signal"] = self._check_signal_exit(ctx, cfg)

        # Final decision as in evaluate()
        results["final"] = self.evaluate(ctx)

        return results

    # ------------------------------------------------------------------ #
    # Strategy implementations
    # ------------------------------------------------------------------ #

    def _check_emergency(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        """
        Account-level protection.

        Triggers when:
          - account_drawdown_pct >= emergency_drawdown_pct, OR
          - realized daily loss reaches a fraction of daily_loss_limit_eur, OR
          - total_open_risk_eur exceeds emergency_max_open_risk_eur.
        """
        triggers = []

        # 1) Equity drawdown
        if ctx.account_drawdown_pct is not None:
            if ctx.account_drawdown_pct >= cfg.emergency_drawdown_pct:
                triggers.append(
                    f"drawdown {ctx.account_drawdown_pct:.3f} "
                    f">= {cfg.emergency_drawdown_pct:.3f}"
                )

        # 2) Daily loss vs limit
        if ctx.daily_loss_eur is not None and ctx.daily_loss_limit_eur is not None:
            if ctx.daily_loss_limit_eur > 0:
                realized_loss = -min(ctx.daily_loss_eur, 0.0)  # positive number for loss
                trigger_loss = (
                    cfg.emergency_daily_loss_buffer_pct * ctx.daily_loss_limit_eur
                )
                if realized_loss >= trigger_loss:
                    triggers.append(
                        f"daily loss €{realized_loss:.2f} "
                        f">= {cfg.emergency_daily_loss_buffer_pct:.2f} * "
                        f"limit €{ctx.daily_loss_limit_eur:.2f}"
                    )

        # 3) Total open risk
        if ctx.total_open_risk_eur is not None:
            if ctx.total_open_risk_eur >= cfg.emergency_max_open_risk_eur:
                triggers.append(
                    f"open risk €{ctx.total_open_risk_eur:.2f} "
                    f">= €{cfg.emergency_max_open_risk_eur:.2f}"
                )

        if not triggers:
            return ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                confidence=0.0,
                urgency=0.0,
                details={},
            )

        return ExitDecision(
            should_exit=True,
            reason=ExitReason.EMERGENCY,
            confidence=0.99,
            urgency=1.0,
            details={
                "triggers": triggers,
                "message": "EMERGENCY EXIT: account-level risk limits reached",
            },
        )

    def _check_soft_stop(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        """Soft stop helper used by both evaluate() and evaluate_all()."""
        if ctx.unrealized_pnl <= -cfg.soft_stop_loss_eur and ctx.signal_against:
            if ctx.signal_strength >= cfg.soft_stop_min_signal:
                base_conf = 0.90
                conf = self._adjust_confidence(base_conf, ctx)
                urg = self._adjust_urgency(0.85, ctx)
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.SOFT_STOP,
                    confidence=conf,
                    urgency=urg,
                    details={
                        "loss": ctx.unrealized_pnl,
                        "threshold": -cfg.soft_stop_loss_eur,
                        "signal_direction": ctx.signal_direction,
                        "signal_strength": ctx.signal_strength,
                        "message": (
                            f"SOFT STOP: Loss €{ctx.unrealized_pnl:.2f} "
                            f"with opposing signal (strength={ctx.signal_strength:.2f})"
                        ),
                    },
                )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.0,
            urgency=0.0,
            details={},
        )

    def _check_time_decay(
        self,
        ctx: PositionContext,
        peak_pnl: float,
        cfg: ExitConfig,
    ) -> ExitDecision:
        """
        Time decay logic:

        - Losing & older than time_decay_hours -> cut the drag (critical)
        - Profitable but stale (old and gave back a large part of peak) -> take profit
        """
        age_h = ctx.age_hours

        if age_h >= cfg.time_decay_hours:
            # 1) Classic time-decay stop: old + losing
            if ctx.unrealized_pnl <= -cfg.time_decay_stop_eur:
                base_conf = 0.85
                conf = self._adjust_confidence(base_conf, ctx)
                urg = self._adjust_urgency(0.80, ctx)
                return ExitDecision(
                    should_exit=True,
                    reason=ExitReason.TIME_DECAY,
                    confidence=conf,
                    urgency=urg,
                    details={
                        "age_hours": age_h,
                        "threshold_hours": cfg.time_decay_hours,
                        "loss": ctx.unrealized_pnl,
                        "loss_threshold": -cfg.time_decay_stop_eur,
                        "message": (
                            f"TIME DECAY (losing): Position {age_h:.1f}h old, "
                            f"loss €{ctx.unrealized_pnl:.2f}"
                        ),
                    },
                )

            # 2) Stale profit: old, was good, now gave back most of peak
            if ctx.is_profitable and peak_pnl > 0:
                retrace_eur = peak_pnl - ctx.unrealized_pnl
                retrace_pct = retrace_eur / peak_pnl if peak_pnl > 0 else 0.0
                # Trigger if position is much older than time_decay_hours and
                # has retraced > 60% of its peak, i.e. profit is fading.
                if age_h >= cfg.time_decay_hours * 1.5 and retrace_pct >= 0.60:
                    base_conf = 0.80
                    conf = self._adjust_confidence(base_conf, ctx)
                    urg = self._adjust_urgency(0.70, ctx)
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.TIME_DECAY,
                        confidence=conf,
                        urgency=urg,
                        details={
                            "age_hours": age_h,
                            "threshold_hours": cfg.time_decay_hours * 1.5,
                            "peak_pnl": peak_pnl,
                            "current_pnl": ctx.unrealized_pnl,
                            "retrace_pct": retrace_pct,
                            "message": (
                                "TIME DECAY (stale profit): Position old, "
                                "profit has retraced >60% of peak"
                            ),
                        },
                    )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.0,
            urgency=0.0,
            details={},
        )

    def _check_trailing_profit(
        self,
        ctx: PositionContext,
        peak_pnl: float,
        cfg: ExitConfig,
    ) -> ExitDecision:
        """
        Check trailing profit stop.

        Improvements:
        - R-based logic: the larger the R-multiple of the trade (peak_pnl /
          hard_stop_loss_eur), the LESS % retrace is allowed (protect big winners).
        - Day-aware tightening: if the day is already nicely positive, tighten
          trailing further to protect realised daily P&L.

        Uses ATR-based trailing if available (smarter), with
        fallback to percentage/EUR-based thresholds.
        """
        # Need minimum peak profit to trail
        if peak_pnl < cfg.trailing_min_peak_eur:
            return ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                confidence=0.0,
                urgency=0.0,
                details={},
            )

        # Compute ATR in EUR if possible
        atr_eur: Optional[float] = None
        if cfg.trailing_use_atr and ctx.atr and ctx.atr > 0 and ctx.lots > 0:
            symbol = ctx.symbol.upper()
            if "XAU" in symbol:
                # Approx: ATR (USD) * 100 oz * lots
                atr_eur = ctx.atr * 100.0 * ctx.lots
            else:
                # Approx: ATR (price units) * 100k contract * lots
                # (assuming quote currency ~ EUR or close enough)
                atr_eur = ctx.atr * 100_000.0 * ctx.lots

        # ─────────────────────────────────────────────────────
        # Activation: only start trailing after decent profit
        # ─────────────────────────────────────────────────────
        activated = False
        activation_method = ""

        if atr_eur is not None:
            # EUR-based OR ATR-multiple-based activation
            if peak_pnl >= cfg.trailing_activation_eur:
                activated = True
                activation_method = "eur"
            elif peak_pnl >= cfg.trailing_activation_atr * atr_eur:
                activated = True
                activation_method = "atr"
        else:
            if peak_pnl >= cfg.trailing_activation_eur:
                activated = True
                activation_method = "eur"

        if not activated:
            return ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                confidence=0.0,
                urgency=0.0,
                details={},
            )

        # ─────────────────────────────────────────────────────
        # Retrace computation
        # ─────────────────────────────────────────────────────
        retrace_eur = peak_pnl - ctx.unrealized_pnl
        retrace_pct = retrace_eur / peak_pnl if peak_pnl > 0 else 0.0

        # ─────────────────────────────────────────────────────
        # R-based tightening:
        #   R = 1   → slightly looser than base (let it breathe)
        #   R = 2   → base retrace_pct
        #   R = 3   → tighter
        #   R >= 4 → even tighter (protect big winners)
        # smaller pct => less giveback
        # ─────────────────────────────────────────────────────
        base_pct = cfg.trailing_retrace_pct
        base_atr_mult = cfg.trailing_retrace_atr

        hard_stop = max(cfg.hard_stop_loss_eur, 1e-6)
        R = peak_pnl / hard_stop

        if R <= 1.0:
            r_factor = 1.1     # small winner near activation → allow a bit more room
        elif R <= 2.0:
            r_factor = 1.0     # normal base behaviour around 2R
        elif R <= 3.0:
            r_factor = 0.8     # tighter for solid winners (2–3R)
        else:
            r_factor = 0.6     # very tight for big runners (3R+)

        # ─────────────────────────────────────────────────────
        # Day-aware tightening:
        # if the day is already nicely positive, protect more.
        # daily_loss_eur is actually signed daily P&L (negative = loss).
        # We normalise vs daily_loss_limit_eur to get a "how good is the day" ratio.
        # ─────────────────────────────────────────────────────
        daily_factor = 1.0
        if (
            ctx.daily_loss_eur is not None
            and ctx.daily_loss_limit_eur is not None
            and ctx.daily_loss_limit_eur > 0
        ):
            daily_pnl = ctx.daily_loss_eur  # positive = profit, negative = loss
            if daily_pnl > 0:
                ratio = max(0.0, min(daily_pnl / ctx.daily_loss_limit_eur, 1.0))
                # shrink thresholds by up to 30% on very strong days
                daily_factor = 1.0 - 0.3 * ratio

        # Keep daily_factor in a sane band
        daily_factor = max(0.7, min(daily_factor, 1.0))

        # Final dynamic thresholds
        dynamic_retrace_pct = base_pct * r_factor * daily_factor
        dynamic_retrace_pct = max(0.05, min(dynamic_retrace_pct, 0.80))  # sanity clamp

        dynamic_atr_mult = base_atr_mult * r_factor * daily_factor
        dynamic_atr_mult = max(0.5, min(dynamic_atr_mult, 3.0))

        # ─────────────────────────────────────────────────────
        # Check retrace thresholds (ATR first, then %)
        # ─────────────────────────────────────────────────────
        should_exit = False
        retrace_method = ""
        effective_atr_threshold_eur = None

        if atr_eur is not None:
            effective_atr_threshold_eur = dynamic_atr_mult * max(atr_eur, 20.0)
            if retrace_eur >= effective_atr_threshold_eur:
                should_exit = True
                retrace_method = "atr"

        # Percentage retrace as backup or primary if ATR not available
        if not should_exit and retrace_pct >= dynamic_retrace_pct:
            should_exit = True
            retrace_method = "pct"

        if should_exit:
            base_conf = 0.85
            conf = self._adjust_confidence(base_conf, ctx)
            urg = self._adjust_urgency(0.75, ctx)
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.TRAILING_PROFIT,
                confidence=conf,
                urgency=urg,
                details={
                    "peak_pnl": peak_pnl,
                    "current_pnl": ctx.unrealized_pnl,
                    "retrace_eur": retrace_eur,
                    "retrace_pct": retrace_pct,
                    "dynamic_retrace_pct": dynamic_retrace_pct,
                    "activation_method": activation_method,
                    "retrace_method": retrace_method,
                    "atr_eur": atr_eur,
                    "dynamic_atr_mult": dynamic_atr_mult,
                    "effective_atr_threshold_eur": effective_atr_threshold_eur,
                    "R_multiple": R,
                    "message": (
                        f"TRAILING TP: Peak €{peak_pnl:.2f} -> "
                        f"€{ctx.unrealized_pnl:.2f} ({retrace_pct:.1%} retrace, "
                        f"limit {dynamic_retrace_pct:.1%}, R={R:.2f})"
                    ),
                },
            )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.0,
            urgency=0.0,
            details={},
        )

    def _check_momentum_exit(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        """
        Momentum-based profit taking: if position is profitable,
        signal is strongly against, and profit exceeds threshold.
        """
        if ctx.is_profitable and ctx.signal_against:
            if ctx.signal_strength >= cfg.momentum_reversal_signal:
                if ctx.unrealized_pnl >= cfg.momentum_exit_profit_eur:
                    base_conf = 0.80
                    conf = self._adjust_confidence(base_conf, ctx)
                    urg = self._adjust_urgency(0.70, ctx)
                    return ExitDecision(
                        should_exit=True,
                        reason=ExitReason.MOMENTUM_EXIT,
                        confidence=conf,
                        urgency=urg,
                        details={
                            "profit": ctx.unrealized_pnl,
                            "signal_strength": ctx.signal_strength,
                            "message": (
                                f"MOMENTUM EXIT: +€{ctx.unrealized_pnl:.2f} "
                                f"with reversal signal (strength={ctx.signal_strength:.2f})"
                            ),
                        },
                    )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.0,
            urgency=0.0,
            details={},
        )

    def _check_signal_exit(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        """Check if agent signal warrants exit (direction flip or weak signal)."""

        # Skip signal-based exit if signal hasn't been calculated yet (startup)
        if not ctx.signal_valid:
            return ExitDecision(
                should_exit=False,
                reason=ExitReason.HOLD,
                confidence=0.0,
                urgency=0.0,
                details={"message": "Signal not yet valid (startup grace period)"},
            )

        # Direction flip - agent wants opposite position
        if ctx.signal_against and ctx.signal_strength >= cfg.signal_direction_weight:
            base_conf = 0.75
            conf = self._adjust_confidence(base_conf, ctx)
            urg = self._adjust_urgency(0.60, ctx)
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.SIGNAL_EXIT,
                confidence=conf,
                urgency=urg,
                details={
                    "position_side": "LONG" if ctx.side > 0 else "SHORT",
                    "signal_direction": "BEARISH" if ctx.signal_direction < 0 else "BULLISH",
                    "signal_strength": ctx.signal_strength,
                    "message": (
                        f"SIGNAL EXIT: Agent signaling "
                        f"{'SHORT' if ctx.signal_direction < 0 else 'LONG'} "
                        f"while position is {'LONG' if ctx.side > 0 else 'SHORT'}"
                    ),
                },
            )

        # Very weak signal - agent lost conviction
        if ctx.signal_strength < cfg.signal_exit_threshold:
            base_conf = 0.70
            conf = self._adjust_confidence(base_conf, ctx)
            urg = self._adjust_urgency(0.50, ctx)
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.SIGNAL_EXIT,
                confidence=conf,
                urgency=urg,
                details={
                    "signal_strength": ctx.signal_strength,
                    "threshold": cfg.signal_exit_threshold,
                    "message": (
                        f"SIGNAL EXIT: Agent signal weak "
                        f"({ctx.signal_strength:.3f} < {cfg.signal_exit_threshold:.3f})"
                    ),
                },
            )

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.0,
            urgency=0.0,
            details={},
        )

    # ------------------------------------------------------------------ #
    # Peak management (external visibility)
    # ------------------------------------------------------------------ #

    def reset_peak(self, symbol: str) -> None:
        """Reset peak tracking when position(s) for a symbol are closed."""
        keys_to_delete = [
            k for k in self._profit_peaks.keys()
            if k == symbol or k.endswith(f"|{symbol}")
        ]
        for k in keys_to_delete:
            self._profit_peaks.pop(k, None)

    def get_peak(self, symbol: str) -> float:
        """
        Get current peak P&L for a symbol.

        If multiple positions exist for the same symbol, returns the maximum.
        """
        peaks = [
            v for k, v in self._profit_peaks.items()
            if k == symbol or k.endswith(f"|{symbol}")
        ]
        return max(peaks) if peaks else 0.0

    def get_all_peaks(self) -> Dict[str, float]:
        """Get all tracked peaks (debugging / monitoring)."""
        return self._profit_peaks.copy()


# Singleton instance for shared use
_exit_engine_instance: Optional[ExitStrategyEngine] = None


def get_exit_engine() -> ExitStrategyEngine:
    """Get or create the singleton exit engine instance."""
    global _exit_engine_instance
    if _exit_engine_instance is None:
        _exit_engine_instance = ExitStrategyEngine()
    return _exit_engine_instance
