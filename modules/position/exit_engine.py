# -------------------------------------------------------------
# File: modules/position/exit_engine.py
# ExitStrategyEngine — Unified exit logic for BOTH systems
#
# SINGLE SOURCE OF TRUTH for exit decisions.
#
# Enhancements (v2.2):
#   - Per-instrument overrides: risk_policy.yaml -> exit_strategies.per_instrument
#   - Hot-reload on YAML mtime (safe, throttled)
#   - Stronger config validation/clamping (prevents “too-loose” drift)
#   - Lifecycle-aware regime mapping (probe/build/ride/defend supported)
#   - Optional ctx.atr_eur support (preferred over heuristics)
#   - More robust peak tracking and stale-peak reset
# -------------------------------------------------------------

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, Optional

# yaml is optional (fallback to defaults if missing)
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# Training mode check - signal exits bypassed during training
try:
    from modules.voting.core.constants import is_training_mode
except ImportError:  # pragma: no cover
    def is_training_mode() -> bool:
        return False


# -----------------------------
# Small safety utilities
# -----------------------------

def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        if math.isnan(x) or math.isinf(x):
            return lo
    except Exception:
        return lo
    return max(lo, min(hi, x))


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if math.isnan(x) or math.isinf(x):
            return default
        return x
    except Exception:
        return default


def _norm_symbol(sym: str) -> str:
    return (sym or "").replace("_", "").replace(".", "").upper()


# -----------------------------
# Public types
# -----------------------------

class ExitReason(Enum):
    HOLD = auto()
    HARD_STOP = auto()
    SOFT_STOP = auto()
    TIME_DECAY = auto()
    TRAILING_PROFIT = auto()
    MOMENTUM_EXIT = auto()
    SIGNAL_EXIT = auto()
    EMERGENCY = auto()
    MANUAL = auto()


@dataclass
class ExitDecision:
    should_exit: bool
    reason: ExitReason
    confidence: float  # 0..1
    urgency: float     # 0..1
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_critical(self) -> bool:
        return self.reason in (ExitReason.HARD_STOP, ExitReason.EMERGENCY, ExitReason.TIME_DECAY)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_exit": self.should_exit,
            "reason": self.reason.name,
            "confidence": float(self.confidence),
            "urgency": float(self.urgency),
            "is_critical": bool(self.is_critical),
            "details": self.details,
        }


@dataclass
class ExitConfig:
    # Hard stop (ALWAYS EXIT)
    hard_stop_loss_eur: float = 150.0

    # Soft stop (loss + opposing signal)
    soft_stop_loss_eur: float = 80.0
    soft_stop_min_signal: float = 0.3

    # Time decay
    time_decay_hours: float = 4.0
    time_decay_stop_eur: float = 60.0

    # Trailing profit
    trailing_activation_eur: float = 100.0
    trailing_activation_atr: float = 2.0
    trailing_retrace_pct: float = 0.30
    trailing_retrace_atr: float = 1.5
    trailing_use_atr: bool = True
    trailing_min_peak_eur: float = 50.0

    # Momentum exit
    momentum_exit_profit_eur: float = 60.0
    momentum_reversal_signal: float = 0.65

    # Signal exit
    signal_exit_threshold: float = 0.10
    signal_direction_weight: float = 0.8

    # Regime scaling
    volatile_regime_tighten: float = 0.7
    ranging_regime_loosen: float = 1.2
    trending_regime_neutral: float = 1.0

    # Emergency / account protection
    emergency_drawdown_pct: float = 0.08
    emergency_daily_loss_buffer_pct: float = 0.9
    emergency_max_open_risk_eur: float = 2000.0

    # Internal safety clamps (not in YAML typically, but safe defaults)
    _min_hard_stop: float = 30.0
    _max_hard_stop: float = 300.0
    _min_trailing_activation: float = 20.0
    _max_trailing_activation: float = 250.0

    def validated(self) -> "ExitConfig":
        """
        Clamp and normalize config so it cannot become dangerously permissive.
        """
        cfg = self

        hard = _clamp(_safe_float(cfg.hard_stop_loss_eur, 150.0), cfg._min_hard_stop, cfg._max_hard_stop)
        soft = _clamp(_safe_float(cfg.soft_stop_loss_eur, 80.0), 10.0, hard * 0.95)
        soft_min_sig = _clamp(_safe_float(cfg.soft_stop_min_signal, 0.3), 0.05, 0.95)

        td_h = _clamp(_safe_float(cfg.time_decay_hours, 4.0), 0.25, 72.0)
        td_stop = _clamp(_safe_float(cfg.time_decay_stop_eur, 60.0), 5.0, hard * 0.95)

        trail_act = _clamp(_safe_float(cfg.trailing_activation_eur, 100.0), cfg._min_trailing_activation, cfg._max_trailing_activation)
        trail_min_peak = _clamp(_safe_float(cfg.trailing_min_peak_eur, 50.0), 5.0, max(10.0, trail_act))

        trail_pct = _clamp(_safe_float(cfg.trailing_retrace_pct, 0.30), 0.10, 0.60)
        trail_atr_mult = _clamp(_safe_float(cfg.trailing_retrace_atr, 1.5), 0.3, 5.0)
        trail_act_atr = _clamp(_safe_float(cfg.trailing_activation_atr, 2.0), 0.5, 10.0)

        mom_profit = _clamp(_safe_float(cfg.momentum_exit_profit_eur, 60.0), 5.0, 10_000.0)
        mom_sig = _clamp(_safe_float(cfg.momentum_reversal_signal, 0.65), 0.2, 0.99)

        sig_thr = _clamp(_safe_float(cfg.signal_exit_threshold, 0.10), 0.01, 0.50)
        sig_flip = _clamp(_safe_float(cfg.signal_direction_weight, 0.8), 0.2, 0.99)

        v_tight = _clamp(_safe_float(cfg.volatile_regime_tighten, 0.7), 0.3, 1.0)
        r_loose = _clamp(_safe_float(cfg.ranging_regime_loosen, 1.2), 1.0, 2.0)
        t_neut = _clamp(_safe_float(cfg.trending_regime_neutral, 1.0), 0.6, 1.4)

        e_dd = _clamp(_safe_float(cfg.emergency_drawdown_pct, 0.08), 0.01, 0.30)
        e_buf = _clamp(_safe_float(cfg.emergency_daily_loss_buffer_pct, 0.9), 0.5, 1.0)
        e_open = _clamp(_safe_float(cfg.emergency_max_open_risk_eur, 2000.0), 100.0, 1_000_000.0)

        return replace(
            cfg,
            hard_stop_loss_eur=hard,
            soft_stop_loss_eur=soft,
            soft_stop_min_signal=soft_min_sig,
            time_decay_hours=td_h,
            time_decay_stop_eur=td_stop,
            trailing_activation_eur=trail_act,
            trailing_min_peak_eur=trail_min_peak,
            trailing_retrace_pct=trail_pct,
            trailing_retrace_atr=trail_atr_mult,
            trailing_activation_atr=trail_act_atr,
            momentum_exit_profit_eur=mom_profit,
            momentum_reversal_signal=mom_sig,
            signal_exit_threshold=sig_thr,
            signal_direction_weight=sig_flip,
            volatile_regime_tighten=v_tight,
            ranging_regime_loosen=r_loose,
            trending_regime_neutral=t_neut,
            emergency_drawdown_pct=e_dd,
            emergency_daily_loss_buffer_pct=e_buf,
            emergency_max_open_risk_eur=e_open,
        )


@dataclass
class PositionContext:
    # Position-level
    symbol: str
    side: int
    unrealized_pnl: float
    peak_pnl: float
    entry_price: float
    current_price: float
    open_time: float
    lots: float = 0.0
    position_id: str = ""

    # Market context
    atr: Optional[float] = None        # ATR in price units
    atr_eur: Optional[float] = None    # Preferred: ATR already converted to EUR
    volatility: float = 0.02
    regime: str = "normal"             # "volatile/ranging/trending/normal/auto" or lifecycle: "probe/build/ride/defend"

    # Signal / committee
    signal_direction: int = 0
    signal_strength: float = 0.0
    signal_valid: bool = False
    consensus_confidence: float = 0.5

    # Account/portfolio context (optional)
    account_drawdown_pct: Optional[float] = None
    daily_loss_eur: Optional[float] = None
    daily_loss_limit_eur: Optional[float] = None
    total_open_risk_eur: Optional[float] = None

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
        if self.signal_direction == 0:
            return False
        return (self.side > 0 and self.signal_direction < 0) or (self.side < 0 and self.signal_direction > 0)

    @property
    def signal_aligns(self) -> bool:
        if self.signal_direction == 0:
            return False
        return (self.side > 0 and self.signal_direction > 0) or (self.side < 0 and self.signal_direction < 0)

    @property
    def pnl_pips(self) -> float:
        if self.entry_price == 0:
            return 0.0
        symbol = (self.symbol or "").upper()
        if "XAU" in symbol:
            pip_size = 0.1
        elif "JPY" in symbol:
            pip_size = 0.01
        else:
            pip_size = 0.0001
        return (self.current_price - self.entry_price) * self.side / pip_size


# -----------------------------
# YAML loading (base + per-instrument overrides)
# -----------------------------

def _risk_policy_path() -> Path:
    return Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"


def load_exit_config() -> ExitConfig:
    """
    Load base exit configuration from risk_policy.yaml.

    Reads:
      - exit_strategies (preferred)
      - smart_position (fallback mapping)
    """
    if yaml is None:
        return ExitConfig().validated()

    config_path = _risk_policy_path()
    try:
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f) or {}

            exit_cfg = policy.get("exit_strategies", {}) or {}
            if not exit_cfg:
                smart = policy.get("smart_position", {}) or {}
                exit_cfg = {
                    "hard_stop_loss_eur": smart.get("hard_stop_loss_eur", 150.0),
                    "soft_stop_loss_eur": smart.get("soft_stop_loss_eur", 80.0),
                    "time_decay_hours": smart.get("time_decay_hours", 4.0),
                    "time_decay_stop_eur": smart.get("time_decay_stop_eur", 60.0),
                    "trailing_activation_eur": smart.get("profit_take_activation_eur", 100.0),
                    "trailing_retrace_pct": smart.get("profit_take_trail_pct", 0.30),
                    "momentum_exit_profit_eur": smart.get("momentum_exit_profit_eur", 60.0),
                    "momentum_reversal_signal": smart.get("reversal_signal_threshold", 0.65),
                }

            cfg = ExitConfig(**{k: v for k, v in exit_cfg.items() if hasattr(ExitConfig, k)})
            return cfg.validated()
    except Exception as e:
        print(f"[ExitEngine] Failed to load config: {e}")

    return ExitConfig().validated()


def load_exit_per_instrument_overrides() -> Dict[str, Dict[str, Any]]:
    """
    Load per-instrument overrides from risk_policy.yaml.

    Supported location:
      risk_policy.yaml -> exit_strategies:
        per_instrument:
          XAUUSD:
            hard_stop_loss_eur: 200
            trailing_activation_eur: 120
            trailing_retrace_pct: 0.22
          EURUSD:
            trailing_use_atr: true
    """
    if yaml is None:
        return {}

    path = _risk_policy_path()
    try:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            policy = yaml.safe_load(f) or {}
        block = (policy.get("exit_strategies", {}) or {}).get("per_instrument", {}) or {}
        if not isinstance(block, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for k, v in block.items():
            if isinstance(v, dict):
                out[_norm_symbol(str(k))] = v
        return out
    except Exception:
        return {}


# -----------------------------
# Engine
# -----------------------------

class ExitStrategyEngine:
    """
    Unified exit strategy engine.

    Evaluates exit strategies in priority order and returns the first triggered exit.
    """

    def __init__(self, config: Optional[ExitConfig] = None):
        self.config: ExitConfig = (config or load_exit_config()).validated()

        # Per-instrument overrides (symbol-normalized keys)
        self._per_instrument: Dict[str, Dict[str, Any]] = load_exit_per_instrument_overrides()
        self._merged_cfg_cache: Dict[str, ExitConfig] = {}

        # Track profit peaks per position or symbol
        self._profit_peaks: Dict[str, float] = {}

        # Hot reload control
        self._cfg_path: Path = _risk_policy_path()
        self._last_mtime: float = 0.0
        self._last_reload_check_ts: float = 0.0
        self._reload_check_interval_sec: float = 3.0  # throttled

        # Initialize mtime baseline
        try:
            if self._cfg_path.exists():
                self._last_mtime = float(self._cfg_path.stat().st_mtime)
        except Exception:
            self._last_mtime = 0.0

    # -------------------------
    # Reload
    # -------------------------

    def _maybe_reload(self) -> None:
        """
        Hot-reload config and per-instrument overrides if YAML changed.
        Safe and throttled; never raises.
        """
        now = time.time()
        if (now - self._last_reload_check_ts) < self._reload_check_interval_sec:
            return
        self._last_reload_check_ts = now

        try:
            if not self._cfg_path.exists():
                return
            mtime = float(self._cfg_path.stat().st_mtime)
            if mtime <= self._last_mtime:
                return
            # Reload
            self.config = load_exit_config().validated()
            self._per_instrument = load_exit_per_instrument_overrides()
            self._merged_cfg_cache.clear()
            self._last_mtime = mtime
        except Exception:
            # Never let reload break trading
            return

    # -------------------------
    # Internal helpers
    # -------------------------

    def _peak_key(self, ctx: PositionContext) -> str:
        return f"{ctx.position_id}|{ctx.symbol}" if ctx.position_id else ctx.symbol

    def _update_peak(self, ctx: PositionContext) -> None:
        key = self._peak_key(ctx)
        current_peak = self._profit_peaks.get(key, ctx.unrealized_pnl)
        self._profit_peaks[key] = max(current_peak, ctx.unrealized_pnl)

    def _effective_peak_pnl(self, ctx: PositionContext) -> float:
        self._update_peak(ctx)
        peak_key = self._peak_key(ctx)
        tracked_peak = self._profit_peaks.get(peak_key, ctx.unrealized_pnl)

        # Stale peak reset protection:
        # if position is very new + near flat, but tracked_peak is huge -> reset.
        if ctx.age_seconds < 300 and abs(ctx.unrealized_pnl) < 50.0 and tracked_peak > 100.0:
            self._profit_peaks[peak_key] = ctx.unrealized_pnl
            tracked_peak = ctx.unrealized_pnl

        return max(_safe_float(ctx.peak_pnl, ctx.unrealized_pnl), _safe_float(tracked_peak, ctx.unrealized_pnl))

    def _adjust_confidence(self, base_conf: float, ctx: PositionContext) -> float:
        factor = 0.5 + 0.5 * _clamp(_safe_float(ctx.consensus_confidence, 0.5), 0.0, 1.0)
        return _clamp(base_conf * factor, 0.0, 1.0)

    def _adjust_urgency(self, base_urg: float, ctx: PositionContext) -> float:
        vol = _clamp(_safe_float(ctx.volatility, 0.02), 0.0, 1.0)
        if vol >= 0.03:
            factor = 1.1
        elif vol <= 0.01:
            factor = 0.9
        else:
            factor = 1.0
        return _clamp(base_urg * factor, 0.0, 1.0)

    def _cfg_for_symbol(self, symbol: str) -> ExitConfig:
        """
        Merge base config + per-instrument overrides (cached).
        """
        sym = _norm_symbol(symbol)
        cached = self._merged_cfg_cache.get(sym)
        if cached is not None:
            return cached

        overrides = self._per_instrument.get(sym, {})
        if not overrides:
            self._merged_cfg_cache[sym] = self.config
            return self.config

        base_dict = self.config.__dict__.copy()
        for k, v in overrides.items():
            if k in base_dict and k.startswith("_") is False:
                base_dict[k] = v
        merged = ExitConfig(**{k: v for k, v in base_dict.items() if hasattr(ExitConfig, k)}).validated()
        self._merged_cfg_cache[sym] = merged
        return merged

    def _map_lifecycle_regime(self, regime: str) -> str:
        """
        Accept lifecycle regimes coming from SmartPositionManager:
          probe/build/ride/defend/exit

        Map them into behavior buckets:
          - defend is special (tighten trailing + earlier activation)
          - probe tightens soft/time-decay slightly (early cut)
          - ride slightly loosens soft/time-decay (let winners breathe)
        """
        r = (regime or "").lower().strip()
        if r in ("probe", "build", "ride", "defend", "exit"):
            return r
        return r or "normal"

    def _regime_adjusted_config(self, ctx: PositionContext, base_cfg: ExitConfig) -> ExitConfig:
        """
        Adjust thresholds based on regime and lifecycle.
        """
        regime = self._map_lifecycle_regime(ctx.regime)

        # Lifecycle: DEFEND = tighten trailing + activate earlier
        if regime == "defend":
            scale = 0.75
            return replace(
                base_cfg,
                trailing_activation_eur=base_cfg.trailing_activation_eur * 0.8,
                trailing_activation_atr=base_cfg.trailing_activation_atr * 0.8,
                trailing_retrace_pct=_clamp(base_cfg.trailing_retrace_pct * scale, 0.10, 0.60),
                trailing_retrace_atr=_clamp(base_cfg.trailing_retrace_atr * scale, 0.3, 5.0),
                trailing_min_peak_eur=base_cfg.trailing_min_peak_eur * 0.8,
                momentum_exit_profit_eur=base_cfg.momentum_exit_profit_eur * 0.8,
                momentum_reversal_signal=_clamp(base_cfg.momentum_reversal_signal * 0.9, 0.2, 0.99),
            ).validated()

        # Lifecycle: PROBE = cut losers a bit sooner (but do not touch HARD_STOP)
        if regime == "probe":
            return replace(
                base_cfg,
                soft_stop_loss_eur=base_cfg.soft_stop_loss_eur * 0.85,
                time_decay_hours=max(0.25, base_cfg.time_decay_hours * 0.6),
                time_decay_stop_eur=base_cfg.time_decay_stop_eur * 0.85,
            ).validated()

        # Lifecycle: RIDE = allow more breathing room
        if regime == "ride":
            return replace(
                base_cfg,
                soft_stop_loss_eur=min(base_cfg.hard_stop_loss_eur * 0.95, base_cfg.soft_stop_loss_eur * 1.10),
                time_decay_hours=min(72.0, base_cfg.time_decay_hours * 1.25),
            ).validated()

        # Classic regimes: auto/normal => infer by volatility
        r = (ctx.regime or "normal").lower()
        if r in ("auto", "normal", ""):
            vol = _clamp(_safe_float(ctx.volatility, 0.02), 0.0, 1.0)
            if vol >= 0.03:
                r = "volatile"
            elif vol <= 0.01:
                r = "ranging"
            else:
                r = "trending"

        if r in ("volatile", "high_volatility"):
            scale = base_cfg.volatile_regime_tighten
        elif r in ("ranging", "sideways", "low_volatility"):
            scale = base_cfg.ranging_regime_loosen
        else:
            scale = base_cfg.trending_regime_neutral

        if abs(scale - 1.0) < 1e-9:
            return base_cfg

        # Scale numeric thresholds where appropriate (do not scale time)
        return replace(
            base_cfg,
            hard_stop_loss_eur=base_cfg.hard_stop_loss_eur * scale,
            soft_stop_loss_eur=base_cfg.soft_stop_loss_eur * scale,
            time_decay_stop_eur=base_cfg.time_decay_stop_eur * scale,
            trailing_retrace_pct=_clamp(base_cfg.trailing_retrace_pct * scale, 0.10, 0.60),
            trailing_retrace_atr=_clamp(base_cfg.trailing_retrace_atr * scale, 0.3, 5.0),
            trailing_min_peak_eur=base_cfg.trailing_min_peak_eur * scale,
            momentum_exit_profit_eur=base_cfg.momentum_exit_profit_eur * scale,
        ).validated()

    def _atr_eur(self, ctx: PositionContext, cfg: ExitConfig) -> Optional[float]:
        """
        Prefer ctx.atr_eur if provided.
        Otherwise fall back to heuristics based on symbol & lot size.
        """
        if not cfg.trailing_use_atr:
            return None

        if ctx.atr_eur is not None and ctx.atr_eur > 0:
            return float(ctx.atr_eur)

        if ctx.atr is None or ctx.atr <= 0 or ctx.lots <= 0:
            return None

        symbol = (ctx.symbol or "").upper()
        atr = float(ctx.atr)

        # Heuristic contract conversions (approximate; best is ctx.atr_eur)
        if "XAU" in symbol:
            return atr * 100.0 * ctx.lots  # 100 oz per lot-ish
        else:
            return atr * 100_000.0 * ctx.lots

    # -------------------------
    # Public API
    # -------------------------

    def evaluate(self, ctx: PositionContext) -> ExitDecision:
        """
        Evaluate all exit strategies for a position.
        """
        self._maybe_reload()

        # Effective peak P&L (internal + external)
        ctx_peak = self._effective_peak_pnl(ctx)

        # Base cfg + per-instrument + regime adjustments
        base_cfg = self._cfg_for_symbol(ctx.symbol)
        cfg = self._regime_adjusted_config(ctx, base_cfg)

        # 0) EMERGENCY
        emergency = self._check_emergency(ctx, cfg)
        if emergency.should_exit:
            return emergency

        # 1) HARD STOP
        if _safe_float(ctx.unrealized_pnl) <= -cfg.hard_stop_loss_eur:
            return ExitDecision(
                should_exit=True,
                reason=ExitReason.HARD_STOP,
                confidence=0.99,
                urgency=1.0,
                details={
                    "loss": float(ctx.unrealized_pnl),
                    "threshold": float(-cfg.hard_stop_loss_eur),
                    "message": f"HARD STOP: Loss €{ctx.unrealized_pnl:.2f} exceeds -€{cfg.hard_stop_loss_eur:.0f}",
                },
            )

        # 2) SOFT STOP
        soft = self._check_soft_stop(ctx, cfg)
        if soft.should_exit:
            return soft

        # 3) TIME DECAY
        td = self._check_time_decay(ctx, ctx_peak, cfg)
        if td.should_exit:
            return td

        # 4) TRAILING PROFIT
        tr = self._check_trailing_profit(ctx, ctx_peak, cfg)
        if tr.should_exit:
            return tr

        # 5) MOMENTUM EXIT
        mo = self._check_momentum_exit(ctx, cfg)
        if mo.should_exit:
            return mo

        # 6) SIGNAL EXIT
        se = self._check_signal_exit(ctx, cfg)
        if se.should_exit:
            return se

        return ExitDecision(
            should_exit=False,
            reason=ExitReason.HOLD,
            confidence=0.60,
            urgency=0.0,
            details={
                "pnl": float(ctx.unrealized_pnl),
                "age_hours": float(ctx.age_hours),
                "signal_direction": int(ctx.signal_direction),
                "peak_pnl": float(ctx_peak),
            },
        )

    def evaluate_all(self, ctx: PositionContext) -> Dict[str, ExitDecision]:
        """
        Diagnostics helper: evaluate all strategies and return their decisions.
        """
        self._maybe_reload()

        ctx_peak = self._effective_peak_pnl(ctx)
        base_cfg = self._cfg_for_symbol(ctx.symbol)
        cfg = self._regime_adjusted_config(ctx, base_cfg)

        results: Dict[str, ExitDecision] = {}
        results["emergency"] = self._check_emergency(ctx, cfg)

        # hard stop (synthetic decision)
        if _safe_float(ctx.unrealized_pnl) <= -cfg.hard_stop_loss_eur:
            results["hard_stop"] = ExitDecision(True, ExitReason.HARD_STOP, 0.99, 1.0, {"loss": float(ctx.unrealized_pnl)})
        else:
            results["hard_stop"] = ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

        results["soft_stop"] = self._check_soft_stop(ctx, cfg)
        results["time_decay"] = self._check_time_decay(ctx, ctx_peak, cfg)
        results["trailing"] = self._check_trailing_profit(ctx, ctx_peak, cfg)
        results["momentum"] = self._check_momentum_exit(ctx, cfg)
        results["signal"] = self._check_signal_exit(ctx, cfg)
        results["final"] = self.evaluate(ctx)
        return results

    # -------------------------
    # Strategy implementations
    # -------------------------

    def _check_emergency(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        triggers = []

        dd = ctx.account_drawdown_pct
        if dd is not None and _safe_float(dd) >= cfg.emergency_drawdown_pct:
            triggers.append(f"drawdown {float(dd):.3f} >= {cfg.emergency_drawdown_pct:.3f}")

        if ctx.daily_loss_eur is not None and ctx.daily_loss_limit_eur is not None:
            limit = _safe_float(ctx.daily_loss_limit_eur, 0.0)
            if limit > 0:
                realized_loss = -min(_safe_float(ctx.daily_loss_eur, 0.0), 0.0)
                trigger_loss = cfg.emergency_daily_loss_buffer_pct * limit
                if realized_loss >= trigger_loss:
                    triggers.append(f"daily loss €{realized_loss:.2f} >= {cfg.emergency_daily_loss_buffer_pct:.2f}*limit €{limit:.2f}")

        if ctx.total_open_risk_eur is not None:
            open_risk = _safe_float(ctx.total_open_risk_eur, 0.0)
            if open_risk >= cfg.emergency_max_open_risk_eur:
                triggers.append(f"open risk €{open_risk:.2f} >= €{cfg.emergency_max_open_risk_eur:.2f}")

        if not triggers:
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

        return ExitDecision(
            should_exit=True,
            reason=ExitReason.EMERGENCY,
            confidence=0.99,
            urgency=1.0,
            details={"triggers": triggers, "message": "EMERGENCY EXIT: account-level risk limits reached"},
        )

    def _check_soft_stop(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        if _safe_float(ctx.unrealized_pnl) <= -cfg.soft_stop_loss_eur and ctx.signal_against:
            if _safe_float(ctx.signal_strength) >= cfg.soft_stop_min_signal:
                conf = self._adjust_confidence(0.90, ctx)
                urg = self._adjust_urgency(0.85, ctx)
                return ExitDecision(
                    True,
                    ExitReason.SOFT_STOP,
                    conf,
                    urg,
                    {
                        "loss": float(ctx.unrealized_pnl),
                        "threshold": float(-cfg.soft_stop_loss_eur),
                        "signal_strength": float(ctx.signal_strength),
                        "message": f"SOFT STOP: Loss €{ctx.unrealized_pnl:.2f} with opposing signal (strength={ctx.signal_strength:.2f})",
                    },
                )
        return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

    def _check_time_decay(self, ctx: PositionContext, peak_pnl: float, cfg: ExitConfig) -> ExitDecision:
        age_h = float(ctx.age_hours)
        if age_h < cfg.time_decay_hours:
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

        # old + losing (critical)
        if _safe_float(ctx.unrealized_pnl) <= -cfg.time_decay_stop_eur:
            conf = self._adjust_confidence(0.85, ctx)
            urg = self._adjust_urgency(0.80, ctx)
            return ExitDecision(
                True,
                ExitReason.TIME_DECAY,
                conf,
                urg,
                {
                    "age_hours": age_h,
                    "threshold_hours": float(cfg.time_decay_hours),
                    "loss": float(ctx.unrealized_pnl),
                    "loss_threshold": float(-cfg.time_decay_stop_eur),
                    "message": f"TIME DECAY (losing): Position {age_h:.1f}h old, loss €{ctx.unrealized_pnl:.2f}",
                },
            )

        # stale profit: very old and gave back most of peak
        if ctx.is_profitable and peak_pnl > 0:
            retrace_eur = peak_pnl - _safe_float(ctx.unrealized_pnl)
            retrace_pct = retrace_eur / peak_pnl if peak_pnl > 0 else 0.0
            if age_h >= cfg.time_decay_hours * 1.5 and retrace_pct >= 0.60:
                conf = self._adjust_confidence(0.80, ctx)
                urg = self._adjust_urgency(0.70, ctx)
                return ExitDecision(
                    True,
                    ExitReason.TIME_DECAY,
                    conf,
                    urg,
                    {
                        "age_hours": age_h,
                        "threshold_hours": float(cfg.time_decay_hours * 1.5),
                        "peak_pnl": float(peak_pnl),
                        "current_pnl": float(ctx.unrealized_pnl),
                        "retrace_pct": float(retrace_pct),
                        "message": "TIME DECAY (stale profit): Position old, profit retraced >60% of peak",
                    },
                )

        return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

    def _check_trailing_profit(self, ctx: PositionContext, peak_pnl: float, cfg: ExitConfig) -> ExitDecision:
        if peak_pnl < cfg.trailing_min_peak_eur:
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

        atr_eur = self._atr_eur(ctx, cfg)

        # Activation
        activated = False
        activation_method = ""
        if atr_eur is not None:
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
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

        retrace_eur = peak_pnl - _safe_float(ctx.unrealized_pnl)
        retrace_pct = retrace_eur / peak_pnl if peak_pnl > 0 else 0.0

        # Profit-tier adaptive trailing
        hard_stop = max(cfg.hard_stop_loss_eur, 1e-6)
        R = peak_pnl / hard_stop

        if peak_pnl < 250.0:
            profit_tier_pct = 0.30
            tier_name = "BREATHING"
        elif peak_pnl < 400.0:
            profit_tier_pct = 0.25
            tier_name = "MODERATE"
        elif peak_pnl < 600.0:
            profit_tier_pct = 0.20
            tier_name = "TIGHT"
        else:
            profit_tier_pct = 0.15
            tier_name = "LOCK-IN"

        if R <= 1.5:
            r_factor = 1.05
        elif R <= 3.0:
            r_factor = 1.0
        else:
            r_factor = 0.90

        # Day-aware tightening: strong day => protect more
        daily_factor = 1.0
        if ctx.daily_loss_eur is not None and ctx.daily_loss_limit_eur is not None:
            limit = _safe_float(ctx.daily_loss_limit_eur, 0.0)
            pnl = _safe_float(ctx.daily_loss_eur, 0.0)  # signed
            if limit > 0 and pnl > 0:
                ratio = _clamp(pnl / limit, 0.0, 1.0)
                daily_factor = 1.0 - 0.3 * ratio
        daily_factor = _clamp(daily_factor, 0.7, 1.0)

        dynamic_retrace_pct = _clamp(profit_tier_pct * r_factor * daily_factor, 0.10, 0.35)
        dynamic_atr_mult = _clamp(cfg.trailing_retrace_atr * r_factor * daily_factor, 0.5, 3.0)

        should_exit = False
        retrace_method = ""
        effective_atr_threshold_eur = None

        if atr_eur is not None:
            effective_atr_threshold_eur = dynamic_atr_mult * max(float(atr_eur), 20.0)
            if retrace_eur >= effective_atr_threshold_eur:
                should_exit = True
                retrace_method = "atr"

        if not should_exit and retrace_pct >= dynamic_retrace_pct:
            should_exit = True
            retrace_method = "pct"

        if should_exit:
            conf = self._adjust_confidence(0.85, ctx)
            urg = self._adjust_urgency(0.75, ctx)
            return ExitDecision(
                True,
                ExitReason.TRAILING_PROFIT,
                conf,
                urg,
                {
                    "peak_pnl": float(peak_pnl),
                    "current_pnl": float(ctx.unrealized_pnl),
                    "retrace_eur": float(retrace_eur),
                    "retrace_pct": float(retrace_pct),
                    "dynamic_retrace_pct": float(dynamic_retrace_pct),
                    "profit_tier": tier_name,
                    "profit_tier_base_pct": float(profit_tier_pct),
                    "activation_method": activation_method,
                    "retrace_method": retrace_method,
                    "atr_eur": float(atr_eur) if atr_eur is not None else None,
                    "dynamic_atr_mult": float(dynamic_atr_mult),
                    "effective_atr_threshold_eur": float(effective_atr_threshold_eur) if effective_atr_threshold_eur is not None else None,
                    "R_multiple": float(R),
                    "message": (
                        f"TRAILING TP [{tier_name}]: Peak €{peak_pnl:.2f} -> "
                        f"€{ctx.unrealized_pnl:.2f} ({retrace_pct:.1%} retrace, "
                        f"limit {dynamic_retrace_pct:.1%}, R={R:.2f})"
                    ),
                },
            )

        return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

    def _check_momentum_exit(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        if ctx.is_profitable and ctx.signal_against:
            if _safe_float(ctx.signal_strength) >= cfg.momentum_reversal_signal:
                if _safe_float(ctx.unrealized_pnl) >= cfg.momentum_exit_profit_eur:
                    conf = self._adjust_confidence(0.80, ctx)
                    urg = self._adjust_urgency(0.70, ctx)
                    return ExitDecision(
                        True,
                        ExitReason.MOMENTUM_EXIT,
                        conf,
                        urg,
                        {
                            "profit": float(ctx.unrealized_pnl),
                            "signal_strength": float(ctx.signal_strength),
                            "message": f"MOMENTUM EXIT: +€{ctx.unrealized_pnl:.2f} with reversal signal (strength={ctx.signal_strength:.2f})",
                        },
                    )
        return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

    def _check_signal_exit(self, ctx: PositionContext, cfg: ExitConfig) -> ExitDecision:
        # Startup grace
        if not bool(ctx.signal_valid):
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {"message": "Signal not yet valid (startup grace period)"})

        # Training bypass
        if is_training_mode():
            return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {"message": "Signal exit bypassed (training mode)"})

        # Direction flip
        if ctx.signal_against and _safe_float(ctx.signal_strength) >= cfg.signal_direction_weight:
            conf = self._adjust_confidence(0.75, ctx)
            urg = self._adjust_urgency(0.60, ctx)
            return ExitDecision(
                True,
                ExitReason.SIGNAL_EXIT,
                conf,
                urg,
                {
                    "position_side": "LONG" if ctx.side > 0 else "SHORT",
                    "signal_direction": "BEARISH" if ctx.signal_direction < 0 else "BULLISH",
                    "signal_strength": float(ctx.signal_strength),
                    "message": (
                        "SIGNAL EXIT: Agent signaling "
                        f"{'SHORT' if ctx.signal_direction < 0 else 'LONG'} "
                        f"while position is {'LONG' if ctx.side > 0 else 'SHORT'}"
                    ),
                },
            )

        # Very weak conviction
        if _safe_float(ctx.signal_strength) < cfg.signal_exit_threshold:
            conf = self._adjust_confidence(0.70, ctx)
            urg = self._adjust_urgency(0.50, ctx)
            return ExitDecision(
                True,
                ExitReason.SIGNAL_EXIT,
                conf,
                urg,
                {
                    "signal_strength": float(ctx.signal_strength),
                    "threshold": float(cfg.signal_exit_threshold),
                    "message": f"SIGNAL EXIT: Agent signal weak ({ctx.signal_strength:.3f} < {cfg.signal_exit_threshold:.3f})",
                },
            )

        return ExitDecision(False, ExitReason.HOLD, 0.0, 0.0, {})

    # -------------------------
    # Peak management
    # -------------------------

    def reset_peak(self, symbol: str) -> None:
        sym = symbol or ""
        keys_to_delete = [k for k in list(self._profit_peaks.keys()) if k == sym or k.endswith(f"|{sym}")]
        for k in keys_to_delete:
            self._profit_peaks.pop(k, None)

    def get_peak(self, symbol: str) -> float:
        sym = symbol or ""
        peaks = [v for k, v in self._profit_peaks.items() if k == sym or k.endswith(f"|{sym}")]
        return max(peaks) if peaks else 0.0

    def get_all_peaks(self) -> Dict[str, float]:
        return self._profit_peaks.copy()


# Singleton
_exit_engine_instance: Optional[ExitStrategyEngine] = None


def get_exit_engine() -> ExitStrategyEngine:
    global _exit_engine_instance
    if _exit_engine_instance is None:
        _exit_engine_instance = ExitStrategyEngine()
    return _exit_engine_instance
