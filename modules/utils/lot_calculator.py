# -------------------------------------------------------------
# File: modules/utils/lot_calculator.py
# UNIFIED LOT SIZE CALCULATOR - Single Source of Truth
#
# All lot sizing in the system MUST go through this module.
# This ensures consistent, synchronized, and dynamic lot calculation
# based on account balance, leverage, volatility, and risk parameters.
# -------------------------------------------------------------

from __future__ import annotations

import yaml
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from enum import Enum

from modules.utils.info_bus import InfoBusManager


class RiskLevel(Enum):
    """Risk levels for position sizing."""
    ULTRA_CONSERVATIVE = "ultra_conservative"  # ~0.15% risk per trade
    CONSERVATIVE = "conservative"              # ~0.25% risk per trade
    MODERATE = "moderate"                      # ~0.5% risk per trade
    AGGRESSIVE = "aggressive"                  # ~0.75% risk per trade
    VERY_AGGRESSIVE = "very_aggressive"        # ~1% risk per trade


@dataclass
class LotSizeConfig:
    """
    Unified lot sizing configuration.

    All lot calculations in the system use these parameters.
    """

    # Account parameters
    account_balance: float = 100_000.0       # Account balance / starting equity
    account_leverage: float = 100.0          # 100:1 leverage
    account_currency: str = "EUR"

    # Risk parameters (as percentage of balance)
    risk_per_trade_pct: float = 0.005        # 0.5% risk per trade (prop firm safe)
    max_risk_per_trade_pct: float = 0.01     # Hard cap: 1% max risk

    # Lot constraints
    min_lot: float = 0.01                    # Broker minimum
    max_lot: float = 10.0                    # Reduced for prop firm safety
    lot_step: float = 0.01                   # Lot increment precision

    # Position limits
    max_positions: int = 2                   # Max concurrent positions (prop firm safe)
    # Fraction of account equity allowed as margin usage
    max_exposure_pct: float = 0.15           # Max 15% of margin used

    # Dynamic adjustments
    volatility_scaling: bool = True          # Scale lots by volatility
    drawdown_scaling: bool = True            # Reduce lots during drawdown
    signal_strength_scaling: bool = True     # Scale by signal confidence

    # Prop firm specific
    prop_firm_mode: bool = True              # Enable prop firm protections
    daily_dd_limit: float = 0.05             # 5% daily drawdown limit
    max_dd_limit: float = 0.10               # 10% max drawdown limit (ACCOUNT CLOSED!)
    daily_dd_safety_buffer: float = 0.008    # Stop at 4.2% (buffer before 5%)
    max_dd_safety_buffer: float = 0.015      # Stop at 8.5% (buffer before 10%)
    emergency_close_all_pct: float = 0.09    # CLOSE ALL at 9% (€1000 buffer before death)

    # Per-instrument contract sizes
    contract_sizes: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 100_000.0,
        "GBPUSD": 100_000.0,
        "USDJPY": 100_000.0,
        "XAUUSD": 100.0,       # Gold: 100 oz per lot
        "XAGUSD": 5000.0,      # Silver: 5000 oz per lot
        "BTCUSD": 1.0,         # Bitcoin: 1 BTC per lot
    })

    # Per-instrument pip values (in account currency per standard lot per pip)
    # Pips here follow standard definitions:
    # - FX: 1 pip = 0.0001 (non-JPY), 0.01 (JPY)
    # - XAU/XAG: 1 "pip" = 0.01 in price
    pip_values: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 10.0,        # €/$10 per pip per lot
        "GBPUSD": 10.0,
        "USDJPY": 9.0,         # Approximate, varies with USD/JPY rate
        "XAUUSD": 1.0,         # €/$1 per 0.01 move per lot (100 oz)
        "XAGUSD": 50.0,        # €/$50 per 0.01 move per lot (5000 oz)
    })

    # Volatility baseline (ATR in PRICE UNITS, not pips)
    volatility_baselines: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.0080,      # ~80 pips daily ATR baseline
        "GBPUSD": 0.0100,      # ~100 pips
        "USDJPY": 0.80,        # ~80 pips
        "XAUUSD": 25.0,        # $25 daily ATR for gold
        "XAGUSD": 0.50,        # $0.50 for silver
    })


class UnifiedLotCalculator:
    """
    ═══════════════════════════════════════════════════════════════════════════
    SINGLE SOURCE OF TRUTH for lot size calculation.
    ═══════════════════════════════════════════════════════════════════════════

    All components (Executor, PositionManager, SmartPositionManager)
    must use this calculator for lot sizing to ensure consistency.

    ARCHITECTURE NOTE:
    ─────────────────────────────────────────────────────────────────────────
    This calculator is responsible for ALL risk-based adjustments to lot size.
    Other components should NOT apply these adjustments themselves to avoid
    double-penalizing:

    ✅ HANDLED HERE (single application):
        - Signal strength scaling
        - Volatility scaling
        - Drawdown scaling
        - DynamicRiskController risk_scale
        - Trading mode multiplier
        - Prime hours boost
        - Market context adjustment (regime stability, theme transition, liquidity)
        - Prop firm headroom reduction

    ❌ NOT HERE (handled elsewhere):
        - Hard VETO checks (Executor checks emergency/critical)
        - Memory VETO (PositionManager checks memory_gate.veto)
        - Gate checks (ArbiterLogic checks GatingResult.gate_passed)

    Key principles:
    1. Risk-based sizing: Fixed % of equity per trade
    2. Volatility adjustment: Higher vol = smaller lots
    3. Signal strength: Stronger signal = larger lots (within limits)
    4. Account leverage: Properly accounts for margin requirements
    5. Drawdown protection: Reduces size during losing streaks
    6. Prop firm safety: Enforces daily/max DD and headroom-based throttling
    ═══════════════════════════════════════════════════════════════════════════
    """

    _instance: Optional["UnifiedLotCalculator"] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_config(config)
        self._bus = None
        self._last_balance_sync = 0.0

    # --------------------------------------------------------------------- #
    # Singleton access
    # --------------------------------------------------------------------- #

    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> "UnifiedLotCalculator":
        """Get singleton instance of the lot calculator."""
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # --------------------------------------------------------------------- #
    # Config loading
    # --------------------------------------------------------------------- #

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> LotSizeConfig:
        """Load configuration from YAML or provided dict."""
        base_config: Dict[str, Any] = {}

        config_path = Path(__file__).parent.parent.parent / "config" / "risk_policy.yaml"
        try:
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}

                lot_config = policy.get("lot_sizing", {}) or {}
                smart_pos = policy.get("smart_position", {}) or {}
                limits = policy.get("limits", {}) or {}
                prop_firm = policy.get("prop_firm", {}) or {}

                # Account / starting balance:
                # Prefer explicit lot_sizing.account_balance, then prop_firm.account_size
                account_balance = lot_config.get(
                    "account_balance",
                    prop_firm.get("account_size", 100_000.0),
                )

                # Max exposure: prefer explicit lot_sizing.max_exposure_pct,
                # then limits.max_exposure_pct, then limits.max_position_size.
                max_exposure_pct = lot_config.get(
                    "max_exposure_pct",
                    limits.get(
                        "max_exposure_pct",
                        limits.get("max_position_size", 0.15),
                    ),
                )

                base_config = {
                    # Account
                    "account_balance": float(account_balance),
                    "account_leverage": float(lot_config.get("account_leverage", 100.0)),

                    # Risk parameters
                    "risk_per_trade_pct": float(lot_config.get("risk_per_trade_pct", 0.005)),
                    "max_risk_per_trade_pct": float(lot_config.get("max_risk_per_trade_pct", 0.01)),

                    # Lot constraints
                    "min_lot": float(lot_config.get("min_lot", 0.01)),
                    "max_lot": float(lot_config.get("max_lot", smart_pos.get("max_lot_size", 10.0))),
                    "lot_step": float(lot_config.get("lot_step", 0.01)),

                    # Position limits
                    "max_positions": int(lot_config.get("max_positions", smart_pos.get("max_total_positions", 2))),
                    "max_exposure_pct": float(max_exposure_pct),

                    # Dynamic scaling
                    "volatility_scaling": bool(lot_config.get("volatility_scaling", True)),
                    "drawdown_scaling": bool(lot_config.get("drawdown_scaling", True)),
                    "signal_strength_scaling": bool(lot_config.get("signal_strength_scaling", True)),

                    # Prop firm settings
                    "prop_firm_mode": bool(prop_firm.get("enabled", True)),
                    "daily_dd_limit": float(prop_firm.get("daily_drawdown_limit", 0.05)),
                    "max_dd_limit": float(prop_firm.get("max_drawdown_limit", 0.10)),
                    "daily_dd_safety_buffer": float(prop_firm.get("daily_dd_safety_buffer", 0.008)),
                    "max_dd_safety_buffer": float(prop_firm.get("max_dd_safety_buffer", 0.015)),
                    # CRITICAL: Close ALL at 9% to protect €1000 buffer before 10%
                    "emergency_close_all_pct": float(prop_firm.get("emergency_close_all_threshold", 0.09)),
                }
        except Exception as e:
            print(f"[LotCalculator] Config load warning: {e}")

        # Override with provided config (tests / special environments)
        if config:
            base_config.update(config)

        return LotSizeConfig(**base_config)

    # --------------------------------------------------------------------- #
    # InfoBus helpers
    # --------------------------------------------------------------------- #

    @property
    def bus(self):
        """Lazy-load InfoBus."""
        if self._bus is None:
            try:
                self._bus = InfoBusManager.get_instance()
            except Exception:
                self._bus = None
        return self._bus

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        return symbol.upper().replace("/", "").replace("_", "")

    @staticmethod
    def _extract_equity_like(data: Any) -> Optional[float]:
        """Try to extract equity/balance from a dict."""
        if not isinstance(data, dict):
            return None

        # Prefer equity-like fields
        for key in ("equity", "net_equity", "account_equity"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)

        # Fallback to balance-like fields
        for key in ("balance", "account_balance"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)

        return None

    def get_current_balance(self) -> float:
        """
        Get current account equity/balance from InfoBus or config default.

        Priority:
        1. Live adapter (equity or balance)
        2. Portfolio metrics
        3. Market state
        4. Environment config (initial_balance)
        5. Config default
        """
        try:
            if self.bus:
                live_status = self.bus.get("live_adapter_status", "LotCalculator", default=None)
                bal = self._extract_equity_like(live_status)
                if bal is not None:
                    return bal

                pm = self.bus.get("portfolio_metrics", "LotCalculator", default=None)
                bal = self._extract_equity_like(pm)
                if bal is not None:
                    return bal

                ms = self.bus.get("market_state", "LotCalculator", default=None)
                bal = self._extract_equity_like(ms)
                if bal is not None:
                    return bal

                env_cfg = self.bus.get("environment_config", "LotCalculator", default=None)
                if isinstance(env_cfg, dict):
                    bal = env_cfg.get("initial_balance")
                    if isinstance(bal, (int, float)) and bal > 0:
                        return float(bal)
        except Exception:
            pass

        return float(self.config.account_balance)

    def get_current_leverage(self) -> float:
        """
        Get current account leverage from live adapter or config default.

        Priority:
        1. Live adapter status (from MT5)
        2. Config default (100.0)
        """
        try:
            if self.bus:
                live_status = self.bus.get("live_adapter_status", "LotCalculator", default=None)
                if isinstance(live_status, dict):
                    leverage = live_status.get("leverage")
                    if isinstance(leverage, (int, float)) and leverage > 0:
                        return float(leverage)
        except Exception:
            pass
        return float(self.config.account_leverage)

    def get_current_drawdown(self) -> float:
        """
        Get current drawdown from InfoBus.

        Expected to be a fraction (e.g. 0.035 = 3.5%) if provided by portfolio_metrics.
        """
        try:
            if self.bus:
                pm = self.bus.get("portfolio_metrics", "LotCalculator", default=None)
                if isinstance(pm, dict):
                    for key in ("drawdown", "current_drawdown", "max_drawdown"):
                        dd = pm.get(key)
                        if isinstance(dd, (int, float)):
                            return float(abs(dd))
        except Exception:
            pass
        return 0.0

    def get_risk_scale(self) -> float:
        """
        Get dynamic risk scale from DynamicRiskController.

        Returns a multiplier 0.1–1.5 based on current market risk assessment.
        """
        try:
            if self.bus:
                risk_scale = self.bus.get("risk_scale", "LotCalculator", default=None)
                if isinstance(risk_scale, (int, float)) and risk_scale > 0:
                    return float(max(0.1, min(1.5, risk_scale)))

                risk_scaling = self.bus.get("risk_scaling", "LotCalculator", default=None)
                if isinstance(risk_scaling, dict):
                    scale = risk_scaling.get("current_risk_scale")
                    if isinstance(scale, (int, float)) and scale > 0:
                        return float(max(0.1, min(1.5, scale)))
        except Exception:
            pass
        return 1.0

    def get_trading_mode_multiplier(self) -> Tuple[float, str]:
        """
        Get risk multiplier from TradingModeManager.

        Returns (multiplier, mode_name) tuple.
        """
        try:
            if self.bus:
                mode_config = self.bus.get("mode_config", "LotCalculator", default=None)
                if isinstance(mode_config, dict):
                    multiplier = mode_config.get("risk_multiplier")
                    if isinstance(multiplier, (int, float)) and multiplier > 0:
                        mode_name = self.bus.get("trading_mode", "LotCalculator", default="normal")
                        return float(max(0.5, min(2.0, multiplier))), str(mode_name or "normal")

                trading_mode = self.bus.get("trading_mode", "LotCalculator", default=None)
                if trading_mode:
                    mode_multipliers = {
                        "safe": 0.5,
                        "normal": 1.0,
                        "aggressive": 1.5,
                        "extreme": 2.0,
                    }
                    multiplier = mode_multipliers.get(str(trading_mode).lower(), 1.0)
                    return float(multiplier), str(trading_mode)
        except Exception:
            pass
        return 1.0, "normal"

    def get_prime_hours_multiplier(self) -> Tuple[float, bool]:
        """
        Get lot size multiplier based on prime trading hours.

        During prime hours (configurable, default 14:00-17:00 local time),
        we boost lot sizes as market quality is highest.

        Returns (multiplier, in_prime_window) tuple.
        """
        try:
            if self.bus:
                # Get seasonality voting proposal which contains trading_window
                seasonality = self.bus.get(
                    "SeasonalityRiskExpert_voting_proposal", "LotCalculator", default=None
                )
                if isinstance(seasonality, dict):
                    trading_window = seasonality.get("trading_window", {})
                    if isinstance(trading_window, dict):
                        in_prime = trading_window.get("in_prime_window", False)
                        lot_mult = trading_window.get("prime_hours_lot_multiplier", 1.0)
                        if in_prime and isinstance(lot_mult, (int, float)) and lot_mult > 1.0:
                            return float(lot_mult), True
        except Exception:
            pass
        return 1.0, False

    def get_market_context_adjustment(self) -> Tuple[float, Dict[str, float]]:
        """
        Get lot size adjustment based on market context from UnifiedMarketModule.

        Uses underutilized market outputs:
        - regime_stability: Lower stability = reduce lot size
        - theme_transition: High transition (choppy market) = reduce lot size
        - regime_accuracy: Lower accuracy = less trust in regime = reduce size
        - liquidity_score: Low liquidity = reduce size to avoid slippage

        Returns (multiplier, details) tuple where multiplier is in [0.5, 1.0].
        """
        details: Dict[str, float] = {
            "regime_stability": 0.5,
            "theme_transition": 0.0,
            "theme_strength": 0.0,
            "regime_accuracy": 0.5,
            "liquidity_score": 0.5,
        }

        try:
            if not self.bus:
                return 1.0, details

            # Fetch regime_stability
            rs = self.bus.get("regime_stability", "LotCalculator", default=None)
            if isinstance(rs, (int, float)):
                details["regime_stability"] = float(max(0.0, min(1.0, rs)))
            elif isinstance(rs, dict):
                rsv = rs.get("value", rs.get("stability", 0.5))
                if isinstance(rsv, (int, float)):
                    details["regime_stability"] = float(max(0.0, min(1.0, rsv)))

            # Fetch theme_transition
            tt = self.bus.get("theme_transition", "LotCalculator", default=None)
            if isinstance(tt, (int, float)):
                details["theme_transition"] = float(max(0.0, min(1.0, tt)))

            # Fetch theme_strength
            ts = self.bus.get("theme_strength", "LotCalculator", default=None)
            if isinstance(ts, (int, float)):
                details["theme_strength"] = float(max(0.0, min(1.0, ts)))

            # Fetch regime_accuracy
            ra = self.bus.get("regime_accuracy", "LotCalculator", default=None)
            if isinstance(ra, dict):
                rav = ra.get("value", ra.get("accuracy", 0.5))
                if isinstance(rav, (int, float)):
                    details["regime_accuracy"] = float(max(0.0, min(1.0, rav)))
            elif isinstance(ra, (int, float)):
                details["regime_accuracy"] = float(max(0.0, min(1.0, ra)))

            # Fetch liquidity_score
            liq = self.bus.get("liquidity_score", "LotCalculator", default=None)
            if isinstance(liq, (int, float)):
                details["liquidity_score"] = float(max(0.0, min(1.0, liq)))
            elif isinstance(liq, dict):
                lv = liq.get("score", liq.get("value", 0.5))
                if isinstance(lv, (int, float)):
                    details["liquidity_score"] = float(max(0.0, min(1.0, lv)))

            # Calculate adjustment factors:

            # 1) Regime stability: [0.7, 1.0]
            # Low stability -> reduce size
            stability_factor = 0.7 + 0.3 * details["regime_stability"]

            # 2) Theme transition: [0.8, 1.0]
            # High transition (choppy) -> reduce size
            # theme_stability = strength penalized by transition rate
            theme_stability = details["theme_strength"] * (1.0 - details["theme_transition"])
            theme_factor = 0.8 + 0.2 * theme_stability

            # 3) Regime accuracy: [0.85, 1.0]
            # Low accuracy -> less trust -> smaller size
            accuracy_factor = 0.85 + 0.15 * details["regime_accuracy"]

            # 4) Liquidity: [0.7, 1.0]
            # Low liquidity -> reduce size to avoid slippage
            liquidity_factor = 0.7 + 0.3 * details["liquidity_score"]

            # Combine multiplicatively
            combined = stability_factor * theme_factor * accuracy_factor * liquidity_factor

            # Clamp to [0.5, 1.0] - never reduce more than 50%, never increase
            adjustment = float(max(0.5, min(1.0, combined)))

            return adjustment, details

        except Exception:
            return 1.0, details

    def get_daily_pnl(self) -> float:
        """Get today's P&L from InfoBus (in account currency)."""
        try:
            if self.bus:
                sm = self.bus.get("session_metrics", "LotCalculator", default=None)
                if isinstance(sm, dict):
                    pnl = sm.get("session_pnl") or sm.get("daily_pnl")
                    if isinstance(pnl, (int, float)):
                        return float(pnl)

                pm = self.bus.get("portfolio_metrics", "LotCalculator", default=None)
                if isinstance(pm, dict):
                    pnl = pm.get("daily_pnl") or pm.get("session_pnl")
                    if isinstance(pnl, (int, float)):
                        return float(pnl)
        except Exception:
            pass
        return 0.0

    def get_starting_balance_for_limits(self) -> float:
        """
        Base equity for total max drawdown checks.

        Priority:
        1. prop_firm_state.initial_balance
        2. Config account_balance
        """
        configured = float(self.config.account_balance)
        live_balance = float(self.get_current_balance() or 0.0)

        # Prefer prop_firm_state but reconcile against live MT5 equity/balance if mismatched.
        try:
            if self.bus:
                pf_state = self.bus.get("prop_firm_state", "LotCalculator", default=None)
                if isinstance(pf_state, dict):
                    base = pf_state.get("initial_balance")
                    if isinstance(base, (int, float)) and base > 0:
                        base_f = float(base)
                        if base_f > 0 and live_balance > 0:
                            ratio = base_f / live_balance
                            if ratio > 1.5 or ratio < 0.67:
                                return live_balance
                        return base_f
        except Exception:
            pass

        # Fallback behavior (live-friendly): if the configured account balance
        # is clearly mismatched with the live account balance/equity, use live.
        if configured > 0 and live_balance > 0:
            ratio = configured / live_balance
            if ratio > 1.5 or ratio < 0.67:
                return live_balance

        return configured

    def get_daily_limit_base(self) -> float:
        """
        Base equity for DAILY drawdown checks.

        Priority:
        1. prop_firm_state.daily_limit_base_equity
        2. prop_firm_state.start_of_day_equity
        3. prop_firm_state.yesterday_close_equity
        4. starting_balance_for_limits()
        """
        try:
            if self.bus:
                pf_state = self.bus.get("prop_firm_state", "LotCalculator", default=None)
                if isinstance(pf_state, dict):
                    for key in (
                        "daily_limit_base_equity",
                        "start_of_day_equity",
                        "yesterday_close_equity",
                    ):
                        base = pf_state.get(key)
                        if isinstance(base, (int, float)) and base > 0:
                            return float(base)
        except Exception:
            pass

        return self.get_starting_balance_for_limits()

    # --------------------------------------------------------------------- #
    # Instrument-specific helpers
    # --------------------------------------------------------------------- #

    def get_contract_size(self, symbol: str) -> float:
        """Get contract size for a symbol."""
        symbol_upper = self._normalize_symbol(symbol)

        for key, size in self.config.contract_sizes.items():
            if self._normalize_symbol(key) == symbol_upper:
                return float(size)

        if "XAU" in symbol_upper or "GOLD" in symbol_upper:
            return 100.0
        if "XAG" in symbol_upper or "SILVER" in symbol_upper:
            return 5000.0
        if "BTC" in symbol_upper or "ETH" in symbol_upper:
            return 1.0

        # Default: Forex standard lot
        return 100_000.0

    def get_pip_value(self, symbol: str) -> float:
        """Get pip value in account currency per standard lot."""
        symbol_upper = self._normalize_symbol(symbol)

        for key, value in self.config.pip_values.items():
            if self._normalize_symbol(key) == symbol_upper:
                return float(value)

        if "XAU" in symbol_upper:
            return 1.0  # $1 per 0.01 move per lot
        if "XAG" in symbol_upper:
            return 50.0
        if "JPY" in symbol_upper:
            return 9.0  # Approximate

        return 10.0  # Standard forex

    def get_pip_size(self, symbol: str) -> float:
        """
        Get pip size in PRICE units.

        - FX non-JPY: 0.0001
        - FX JPY pairs: 0.01
        - Gold/Silver (XAU/XAG): 0.01
        """
        symbol_upper = self._normalize_symbol(symbol)

        if "JPY" in symbol_upper:
            return 0.01
        if "XAU" in symbol_upper or "XAG" in symbol_upper:
            return 0.01
        return 0.0001

    def get_volatility_baseline(self, symbol: str) -> float:
        """Get volatility baseline (ATR in PRICE units) for a symbol."""
        symbol_upper = self._normalize_symbol(symbol)

        for key, value in self.config.volatility_baselines.items():
            if self._normalize_symbol(key) == symbol_upper:
                return float(value)

        if "XAU" in symbol_upper:
            return 25.0
        if "JPY" in symbol_upper:
            return 0.80

        return 0.0080  # ~80 pips for standard FX

    # --------------------------------------------------------------------- #
    # Prop firm limits
    # --------------------------------------------------------------------- #

    def check_prop_firm_limits(self) -> Dict[str, Any]:
        """
        Check if prop firm limits are being approached.

        Returns status dict with:
        - can_trade: bool - whether new trades are allowed
        - daily_dd_used: float - daily drawdown used (%)
        - max_dd_used: float - total drawdown used (%)
        - daily_dd_remaining: float
        - max_dd_remaining: float
        - warnings: list[str]
        - must_close_all: bool - emergency close required
        """
        if not self.config.prop_firm_mode:
            return {
                "can_trade": True,
                "warnings": [],
                "must_close_all": False,
                "daily_dd_used": 0.0,
                "max_dd_used": 0.0,
                "daily_dd_remaining": 1.0,
                "max_dd_remaining": 1.0,
            }

        result: Dict[str, Any] = {
            "can_trade": True,
            "daily_dd_used": 0.0,
            "max_dd_used": 0.0,
            "daily_dd_remaining": self.config.daily_dd_limit,
            "max_dd_remaining": self.config.max_dd_limit,
            "warnings": [],
            "must_close_all": False,
        }

        balance = self.get_current_balance()
        starting_balance = self.get_starting_balance_for_limits()
        daily_base = self.get_daily_limit_base()

        # Total max drawdown from starting balance
        if starting_balance > 0:
            total_dd = (starting_balance - balance) / starting_balance
        else:
            total_dd = 0.0

        total_dd = max(0.0, total_dd)
        result["max_dd_used"] = total_dd
        result["max_dd_remaining"] = max(0.0, self.config.max_dd_limit - total_dd)

        # Daily drawdown from today's PnL
        daily_pnl = self.get_daily_pnl()
        if daily_pnl < 0 and daily_base > 0:
            daily_dd = abs(daily_pnl) / daily_base
        else:
            daily_dd = 0.0

        result["daily_dd_used"] = daily_dd
        result["daily_dd_remaining"] = max(0.0, self.config.daily_dd_limit - daily_dd)

        # Effective (buffered) limits
        effective_daily_limit = self.config.daily_dd_limit - self.config.daily_dd_safety_buffer
        effective_max_limit = self.config.max_dd_limit - self.config.max_dd_safety_buffer

        # Daily limit checks
        if daily_dd >= self.config.daily_dd_limit:
            result["can_trade"] = False
            result["must_close_all"] = True
            result["warnings"].append(
                f"CRITICAL: Daily DD limit breached ({daily_dd:.2%} >= {self.config.daily_dd_limit:.2%})"
            )
        elif daily_dd >= effective_daily_limit:
            result["can_trade"] = False
            result["warnings"].append(
                f"STOP: Approaching daily DD limit ({daily_dd:.2%} / {self.config.daily_dd_limit:.2%})"
            )
        elif daily_dd >= effective_daily_limit * 0.8:
            result["warnings"].append(
                f"WARNING: Daily DD at {daily_dd:.2%} (limit: {self.config.daily_dd_limit:.2%})"
            )

        # Max drawdown checks
        # CRITICAL: 10% = ACCOUNT CLOSED! We have 3 thresholds:
        # 1) 8.5% (effective_max_limit) = Stop new trades
        # 2) 9.0% (emergency_close_all_pct) = CLOSE ALL POSITIONS
        # 3) 10% (max_dd_limit) = Account death (should never reach)
        
        if total_dd >= self.config.emergency_close_all_pct:
            # EMERGENCY: Close everything at 9%
            result["can_trade"] = False
            result["must_close_all"] = True
            result["warnings"].append(
                f"🚨 EMERGENCY: DD at {total_dd:.2%} >= {self.config.emergency_close_all_pct:.2%} - CLOSE ALL!"
            )
        elif total_dd >= self.config.max_dd_limit:
            result["can_trade"] = False
            result["must_close_all"] = True
            result["warnings"].append(
                f"CRITICAL: Max DD limit breached ({total_dd:.2%} >= {self.config.max_dd_limit:.2%})"
            )
        elif total_dd >= effective_max_limit:
            result["can_trade"] = False
            result["warnings"].append(
                f"STOP: Approaching max DD limit ({total_dd:.2%} / {self.config.max_dd_limit:.2%})"
            )
        elif total_dd >= effective_max_limit * 0.8:
            result["warnings"].append(
                f"WARNING: Max DD at {total_dd:.2%} (limit: {self.config.max_dd_limit:.2%})"
            )

        return result

    def get_prop_firm_lot_reduction(self) -> float:
        """
        Get lot reduction factor based on proximity to prop firm limits.

        Returns multiplier 0.0–1.0 to apply to lot size.
        """
        if not self.config.prop_firm_mode:
            return 1.0

        limits = self.check_prop_firm_limits()

        if not limits["can_trade"]:
            return 0.0

        daily_remaining = limits["daily_dd_remaining"]
        max_remaining = limits["max_dd_remaining"]

        daily_headroom = (
            daily_remaining / self.config.daily_dd_limit if self.config.daily_dd_limit > 0 else 1.0
        )
        max_headroom = (
            max_remaining / self.config.max_dd_limit if self.config.max_dd_limit > 0 else 1.0
        )

        headroom = min(daily_headroom, max_headroom)

        # At 100% headroom: full size
        # At 50% headroom: 50% size
        # At 20% headroom: 20% size (clipped to 0.1 minimum)
        return max(0.1, min(1.0, headroom))

    # --------------------------------------------------------------------- #
    # Core lot calculation
    # --------------------------------------------------------------------- #

    def calculate_lots(
        self,
        symbol: str,
        signal_strength: float = 1.0,
        stop_loss_pips: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        override_balance: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal lot size based on risk parameters.

        Args:
            symbol: Trading symbol (e.g., "EURUSD", "XAUUSD").
            signal_strength: Signal confidence 0.0–1.0 (affects final size).
            stop_loss_pips: Stop loss distance in pips (as defined by get_pip_size()).
                            If None, will be derived from volatility or defaults.
            volatility: Current volatility (ATR in PRICE units, not pips).
            risk_level: Risk level enum (maps to % of equity).
            override_balance: Override account balance/equity (for testing).

        Returns:
            Tuple of (lot_size, calculation_details).
        """
        details: Dict[str, Any] = {
            "symbol": symbol,
            "method": "risk_based",
            "adjustments": [],
            "prop_firm_mode": self.config.prop_firm_mode,
        }

        # 0. Prop firm check
        if self.config.prop_firm_mode:
            prop_status = self.check_prop_firm_limits()
            details["prop_firm_status"] = prop_status

            if not prop_status["can_trade"]:
                details["blocked"] = True
                details["block_reason"] = prop_status["warnings"]
                details["final_lots"] = 0.0
                return 0.0, details

            if prop_status["warnings"]:
                details["prop_firm_warnings"] = prop_status["warnings"]

        # 1. Effective balance/equity
        balance = self.get_current_balance() if override_balance is None else float(override_balance)
        details["balance"] = balance

        # 2. Risk % based on level (prop-firm safe mapping)
        risk_pct_map = {
            RiskLevel.ULTRA_CONSERVATIVE: 0.0015,  # 0.15%
            RiskLevel.CONSERVATIVE: 0.0025,        # 0.25%
            RiskLevel.MODERATE: 0.005,             # 0.5%
            RiskLevel.AGGRESSIVE: 0.0075,          # 0.75%
            RiskLevel.VERY_AGGRESSIVE: 0.01,       # 1.0%
        }
        base_risk_pct = risk_pct_map.get(risk_level, self.config.risk_per_trade_pct)

        risk_pct = min(base_risk_pct, self.config.max_risk_per_trade_pct)
        details["risk_pct"] = risk_pct
        details["risk_level"] = risk_level.value

        # 3. Risk amount in account currency
        risk_amount = balance * risk_pct
        details["risk_amount"] = risk_amount

        # 4. Pip value & pip size
        pip_value = self.get_pip_value(symbol)
        pip_size = self.get_pip_size(symbol)
        details["pip_value_per_lot"] = pip_value
        details["pip_size"] = pip_size

        # 5. Effective stop loss distance (in pips)
        symbol_upper = self._normalize_symbol(symbol)

        if stop_loss_pips is None:
            if volatility is not None and volatility > 0:
                # volatility and baseline in PRICE units
                vol_baseline_price = self.get_volatility_baseline(symbol)
                current_price_range = float(volatility)
                baseline_price_range = float(vol_baseline_price) if vol_baseline_price > 0 else current_price_range

                # Stop loss in PRICE units
                stop_loss_price = max(current_price_range * 1.5, baseline_price_range * 0.5)
            else:
                # Instrument-specific default stops in PRICE units
                if "XAU" in symbol_upper:
                    # Default ~ $10 SL, consistent with sl_tp_settings (1000 pips at 0.01)
                    stop_loss_price = 10.0
                elif "XAG" in symbol_upper:
                    stop_loss_price = 0.30
                else:
                    # FX: default 100 pips (~0.0100)
                    stop_loss_price = 0.0100

            # Convert price distance to pips
            stop_loss_pips = max(1.0, stop_loss_price / pip_size)
        else:
            stop_loss_pips = max(1.0, float(stop_loss_pips))

        details["stop_loss_pips"] = stop_loss_pips

        # 6. Base lot size: risk_amount / (SL * pip_value)
        if stop_loss_pips > 0 and pip_value > 0:
            base_lots = risk_amount / (stop_loss_pips * pip_value)
        else:
            # Fallback: leverage-based calculation
            contract_size = self.get_contract_size(symbol)
            leverage = self.get_current_leverage()  # Use live leverage from MT5
            max_margin_lots = (balance * leverage) / contract_size
            base_lots = max_margin_lots * risk_pct * 10.0  # scaled heuristic
        details["base_lots"] = base_lots

        # 7. Signal strength scaling
        if self.config.signal_strength_scaling:
            sig = max(0.1, min(1.0, float(signal_strength)))
            signal_multiplier = 0.5 + (sig * 0.5)  # 0.5x–1.0x
            base_lots *= signal_multiplier
            details["adjustments"].append(f"signal_strength={sig:.2f}→{signal_multiplier:.2f}x")

        # 8. Volatility scaling (ATR in PRICE units)
        if self.config.volatility_scaling and volatility is not None and volatility > 0:
            vol_baseline_price = self.get_volatility_baseline(symbol)
            if vol_baseline_price > 0:
                vol_ratio = float(volatility) / float(vol_baseline_price)
                # High vol => smaller size, low vol => larger (capped)
                vol_multiplier = max(0.5, min(1.5, 1.0 / max(vol_ratio, 1e-6)))
                base_lots *= vol_multiplier
                details["adjustments"].append(f"volatility={volatility:.4f}→{vol_multiplier:.2f}x")

        # 9. Drawdown scaling
        if self.config.drawdown_scaling:
            drawdown = self.get_current_drawdown()  # expected as fraction
            if drawdown > 0.02:  # more than 2% drawdown
                # At 10% DD, size is halved; floor at 0.3x
                dd_multiplier = max(0.3, 1.0 - (drawdown * 5.0))
                base_lots *= dd_multiplier
                details["adjustments"].append(f"drawdown={drawdown:.2%}→{dd_multiplier:.2f}x")

        # 10. DynamicRiskController scale
        risk_scale = self.get_risk_scale()
        if risk_scale != 1.0:
            base_lots *= risk_scale
            details["adjustments"].append(f"risk_controller_scale={risk_scale:.2f}x")
            details["risk_scale"] = risk_scale

        # 11. Trading mode multiplier
        mode_multiplier, mode_name = self.get_trading_mode_multiplier()
        if mode_multiplier != 1.0:
            base_lots *= mode_multiplier
            details["adjustments"].append(f"trading_mode={mode_name}→{mode_multiplier:.2f}x")
        details["trading_mode"] = mode_name
        details["trading_mode_multiplier"] = mode_multiplier

        # 11b. Prime hours lot boost (best market quality window)
        prime_multiplier, in_prime = self.get_prime_hours_multiplier()
        if in_prime and prime_multiplier > 1.0:
            base_lots *= prime_multiplier
            details["adjustments"].append(f"prime_hours_boost→{prime_multiplier:.2f}x")
        details["in_prime_window"] = in_prime
        details["prime_hours_multiplier"] = prime_multiplier

        # 11c. Market context adjustment (regime stability, theme transition, liquidity)
        # Uses underutilized outputs from UnifiedMarketModule
        market_ctx_multiplier, market_ctx_details = self.get_market_context_adjustment()
        if market_ctx_multiplier < 1.0:
            base_lots *= market_ctx_multiplier
            details["adjustments"].append(f"market_context={market_ctx_multiplier:.2f}x")
        details["market_context"] = market_ctx_details
        details["market_context_multiplier"] = market_ctx_multiplier

        # 12. Prop firm headroom reduction
        # CRITICAL: If prop firm limits are breached, return 0 lots to block trading
        if self.config.prop_firm_mode:
            prop_reduction = self.get_prop_firm_lot_reduction()
            if prop_reduction == 0.0:
                # Prop firm limits breached - BLOCK ALL TRADING
                details["adjustments"].append("prop_firm_BLOCKED")
                details["prop_firm_reduction"] = 0.0
                details["final_lots"] = 0.0
                details["blocked_reason"] = "prop_firm_limits_breached"
                return 0.0, details
            elif prop_reduction < 1.0:
                base_lots *= prop_reduction
                details["adjustments"].append(f"prop_firm_headroom={prop_reduction:.2f}x")
                details["prop_firm_reduction"] = prop_reduction

        # 13. Round to lot step
        lots = self._round_to_step(base_lots, self.config.lot_step)

        # 14. Apply min/max lot constraints
        lots = max(self.config.min_lot, min(lots, self.config.max_lot))

        # 15. Margin / exposure check
        contract_size = self.get_contract_size(symbol)
        leverage = self.get_current_leverage()  # Use live leverage from MT5
        max_margin = balance * self.config.max_exposure_pct
        margin_required = (lots * contract_size) / leverage

        if margin_required > max_margin:
            # Reduce lots to fit margin constraint
            allowed_lots = (max_margin * leverage) / contract_size
            allowed_lots = self._round_to_step(allowed_lots, self.config.lot_step, mode="down")
            allowed_lots = max(self.config.min_lot, min(allowed_lots, self.config.max_lot))
            lots = allowed_lots
            margin_required = (lots * contract_size) / leverage
            details["adjustments"].append(f"margin_cap→{lots:.2f}")

        details["final_lots"] = lots
        details["margin_required"] = margin_required
        details["margin_available"] = max_margin

        return lots, details

    def calculate_lots_simple(
        self,
        symbol: str,
        signal_strength: float = 1.0,
        volatility: Optional[float] = None,
    ) -> float:
        """
        Simplified lot calculation for quick use.

        Returns just the lot size without details.
        """
        lots, _ = self.calculate_lots(
            symbol=symbol,
            signal_strength=signal_strength,
            volatility=volatility,
        )
        return lots

    def calculate_lots_from_size_eur(
        self,
        symbol: str,
        size_eur: float,
        current_price: float,
    ) -> float:
        """
        Convert EUR notional size to lots.

        Used for backward compatibility with systems that specify size in EUR.
        """
        if size_eur <= 0 or current_price <= 0:
            return self.config.min_lot

        contract_size = self.get_contract_size(symbol)
        units = size_eur / current_price
        lots = units / contract_size

        lots = self._round_to_step(lots, self.config.lot_step)
        lots = max(self.config.min_lot, min(lots, self.config.max_lot))

        return lots

    def get_max_lots_for_balance(
        self,
        symbol: str,
        balance: Optional[float] = None,
    ) -> float:
        """Calculate maximum lots possible given balance and leverage."""
        if balance is None:
            balance = self.get_current_balance()

        contract_size = self.get_contract_size(symbol)
        leverage = self.get_current_leverage()  # Use live leverage from MT5

        leverage_max = (balance * leverage) / contract_size

        exposure_max = (
            balance * self.config.max_exposure_pct * leverage
        ) / contract_size

        max_lots = min(leverage_max, exposure_max, self.config.max_lot)

        return self._round_to_step(max_lots, self.config.lot_step, mode="down")

    @staticmethod
    def _round_to_step(
        value: float,
        step: float,
        mode: str = "nearest",
    ) -> float:
        """Round value to step increment."""
        if step <= 0:
            return value

        if mode == "down":
            return math.floor(value / step) * step
        elif mode == "up":
            return math.ceil(value / step) * step
        else:
            return round(value / step) * step

    def publish_lot_config_to_bus(self) -> None:
        """Publish current lot configuration to InfoBus for other modules."""
        if not self.bus:
            return

        try:
            config_data = {
                "account_balance": self.get_current_balance(),
                "account_leverage": self.get_current_leverage(),  # Use live leverage
                "risk_per_trade_pct": self.config.risk_per_trade_pct,
                "min_lot": self.config.min_lot,
                "max_lot": self.config.max_lot,
                "lot_step": self.config.lot_step,
                "max_positions": self.config.max_positions,
                "max_exposure_pct": self.config.max_exposure_pct,
            }

            self.bus.set(
                "lot_config",
                config_data,
                module="LotCalculator",
                thesis="Unified lot sizing configuration",
            )
        except Exception:
            pass


# Convenience function for quick access
def calculate_lots(
    symbol: str,
    signal_strength: float = 1.0,
    stop_loss_pips: Optional[float] = None,
    volatility: Optional[float] = None,
    risk_level: RiskLevel = RiskLevel.MODERATE,
) -> float:
    """
    Quick access to lot calculation.

    Usage:
        from modules.utils.lot_calculator import calculate_lots
        lots = calculate_lots("EURUSD", signal_strength=0.8)
    """
    calculator = UnifiedLotCalculator.get_instance()
    lots, _ = calculator.calculate_lots(
        symbol=symbol,
        signal_strength=signal_strength,
        stop_loss_pips=stop_loss_pips,
        volatility=volatility,
        risk_level=risk_level,
    )
    return lots
