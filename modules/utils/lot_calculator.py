

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from modules.utils.info_bus import InfoBusManager


class RiskLevel(Enum):
    ULTRA_CONSERVATIVE = "ultra_conservative"
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"


@dataclass
class LotSizeConfig:


    account_balance: float = 100_000.0
    account_leverage: float = 100.0
    account_currency: str = "EUR"


    risk_per_trade_pct: float = 0.005
    max_risk_per_trade_pct: float = 0.01


    min_lot: float = 0.01
    max_lot: float = 10.0
    lot_step: float = 0.01


    max_positions: int = 2

    max_exposure_pct: float = 0.15


    volatility_scaling: bool = True
    drawdown_scaling: bool = True
    signal_strength_scaling: bool = True


    prop_firm_mode: bool = True
    daily_dd_limit: float = 0.05
    max_dd_limit: float = 0.10
    daily_dd_safety_buffer: float = 0.008
    max_dd_safety_buffer: float = 0.015
    emergency_close_all_pct: float = 0.09


    contract_sizes: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 100_000.0,
        "GBPUSD": 100_000.0,
        "USDJPY": 100_000.0,
        "XAUUSD": 100.0,
        "XAGUSD": 5000.0,
        "BTCUSD": 1.0,
    })


    pip_values: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 10.0,
        "GBPUSD": 10.0,
        "USDJPY": 9.0,
        "XAUUSD": 1.0,
        "XAGUSD": 50.0,
    })


    volatility_baselines: Dict[str, float] = field(default_factory=lambda: {
        "EURUSD": 0.0080,
        "GBPUSD": 0.0100,
        "USDJPY": 0.80,
        "XAUUSD": 25.0,
        "XAGUSD": 0.50,
    })


class UnifiedLotCalculator:

    _instance: Optional["UnifiedLotCalculator"] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._load_config(config)
        self._bus = None
        self._last_balance_sync = 0.0


    @classmethod
    def get_instance(cls, config: Optional[Dict[str, Any]] = None) -> "UnifiedLotCalculator":
        if cls._instance is None:
            cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        cls._instance = None


    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> LotSizeConfig:
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


                account_balance = lot_config.get(
                    "account_balance",
                    prop_firm.get("account_size", 100_000.0),
                )


                max_exposure_pct = lot_config.get(
                    "max_exposure_pct",
                    limits.get(
                        "max_exposure_pct",
                        limits.get("max_position_size", 0.15),
                    ),
                )

                base_config = {

                    "account_balance": float(account_balance),
                    "account_leverage": float(lot_config.get("account_leverage", 100.0)),


                    "risk_per_trade_pct": float(lot_config.get("risk_per_trade_pct", 0.005)),
                    "max_risk_per_trade_pct": float(lot_config.get("max_risk_per_trade_pct", 0.01)),


                    "min_lot": float(lot_config.get("min_lot", 0.01)),
                    "max_lot": float(lot_config.get("max_lot", smart_pos.get("max_lot_size", 10.0))),
                    "lot_step": float(lot_config.get("lot_step", 0.01)),


                    "max_positions": int(lot_config.get("max_positions", smart_pos.get("max_total_positions", 2))),
                    "max_exposure_pct": float(max_exposure_pct),


                    "volatility_scaling": bool(lot_config.get("volatility_scaling", True)),
                    "drawdown_scaling": bool(lot_config.get("drawdown_scaling", True)),
                    "signal_strength_scaling": bool(lot_config.get("signal_strength_scaling", True)),


                    "prop_firm_mode": bool(prop_firm.get("enabled", True)),
                    "daily_dd_limit": float(prop_firm.get("daily_drawdown_limit", 0.05)),
                    "max_dd_limit": float(prop_firm.get("max_drawdown_limit", 0.10)),
                    "daily_dd_safety_buffer": float(prop_firm.get("daily_dd_safety_buffer", 0.008)),
                    "max_dd_safety_buffer": float(prop_firm.get("max_dd_safety_buffer", 0.015)),

                    "emergency_close_all_pct": float(prop_firm.get("emergency_close_all_threshold", 0.09)),
                }
        except Exception as e:
            print(f"[LotCalculator] Config load warning: {e}")


        if config:
            base_config.update(config)

        return LotSizeConfig(**base_config)


    @property
    def bus(self):
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
        if not isinstance(data, dict):
            return None


        for key in ("equity", "net_equity", "account_equity"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)


        for key in ("balance", "account_balance"):
            val = data.get(key)
            if isinstance(val, (int, float)) and val > 0:
                return float(val)

        return None

    def get_current_balance(self) -> float:
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
        try:
            if self.bus:

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


            rs = self.bus.get("regime_stability", "LotCalculator", default=None)
            if isinstance(rs, (int, float)):
                details["regime_stability"] = float(max(0.0, min(1.0, rs)))
            elif isinstance(rs, dict):
                rsv = rs.get("value", rs.get("stability", 0.5))
                if isinstance(rsv, (int, float)):
                    details["regime_stability"] = float(max(0.0, min(1.0, rsv)))


            tt = self.bus.get("theme_transition", "LotCalculator", default=None)
            if isinstance(tt, (int, float)):
                details["theme_transition"] = float(max(0.0, min(1.0, tt)))


            ts = self.bus.get("theme_strength", "LotCalculator", default=None)
            if isinstance(ts, (int, float)):
                details["theme_strength"] = float(max(0.0, min(1.0, ts)))


            ra = self.bus.get("regime_accuracy", "LotCalculator", default=None)
            if isinstance(ra, dict):
                rav = ra.get("value", ra.get("accuracy", 0.5))
                if isinstance(rav, (int, float)):
                    details["regime_accuracy"] = float(max(0.0, min(1.0, rav)))
            elif isinstance(ra, (int, float)):
                details["regime_accuracy"] = float(max(0.0, min(1.0, ra)))


            liq = self.bus.get("liquidity_score", "LotCalculator", default=None)
            if isinstance(liq, (int, float)):
                details["liquidity_score"] = float(max(0.0, min(1.0, liq)))
            elif isinstance(liq, dict):
                lv = liq.get("score", liq.get("value", 0.5))
                if isinstance(lv, (int, float)):
                    details["liquidity_score"] = float(max(0.0, min(1.0, lv)))


            stability_factor = 0.7 + 0.3 * details["regime_stability"]


            theme_stability = details["theme_strength"] * (1.0 - details["theme_transition"])
            theme_factor = 0.8 + 0.2 * theme_stability


            accuracy_factor = 0.85 + 0.15 * details["regime_accuracy"]


            liquidity_factor = 0.7 + 0.3 * details["liquidity_score"]


            combined = stability_factor * theme_factor * accuracy_factor * liquidity_factor


            adjustment = float(max(0.5, min(1.0, combined)))

            return adjustment, details

        except Exception:
            return 1.0, details

    def get_daily_pnl(self) -> float:
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
        try:
            if self.bus:
                pf_state = self.bus.get("prop_firm_state", "LotCalculator", default=None)
                if isinstance(pf_state, dict):
                    base = pf_state.get("initial_balance")
                    if isinstance(base, (int, float)) and base > 0:
                        return float(base)
        except Exception:
            pass
        return float(self.config.account_balance)

    def get_daily_limit_base(self) -> float:
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


    def get_contract_size(self, symbol: str) -> float:
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


        return 100_000.0

    def get_pip_value(self, symbol: str) -> float:
        symbol_upper = self._normalize_symbol(symbol)

        for key, value in self.config.pip_values.items():
            if self._normalize_symbol(key) == symbol_upper:
                return float(value)

        if "XAU" in symbol_upper:
            return 1.0
        if "XAG" in symbol_upper:
            return 50.0
        if "JPY" in symbol_upper:
            return 9.0

        return 10.0

    def get_pip_size(self, symbol: str) -> float:
        symbol_upper = self._normalize_symbol(symbol)

        if "JPY" in symbol_upper:
            return 0.01
        if "XAU" in symbol_upper or "XAG" in symbol_upper:
            return 0.01
        return 0.0001

    def get_volatility_baseline(self, symbol: str) -> float:
        symbol_upper = self._normalize_symbol(symbol)

        for key, value in self.config.volatility_baselines.items():
            if self._normalize_symbol(key) == symbol_upper:
                return float(value)

        if "XAU" in symbol_upper:
            return 25.0
        if "JPY" in symbol_upper:
            return 0.80

        return 0.0080


    def check_prop_firm_limits(self) -> Dict[str, Any]:
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


        if starting_balance > 0:
            total_dd = (starting_balance - balance) / starting_balance
        else:
            total_dd = 0.0

        total_dd = max(0.0, total_dd)
        result["max_dd_used"] = total_dd
        result["max_dd_remaining"] = max(0.0, self.config.max_dd_limit - total_dd)


        daily_pnl = self.get_daily_pnl()
        if daily_pnl < 0 and daily_base > 0:
            daily_dd = abs(daily_pnl) / daily_base
        else:
            daily_dd = 0.0

        result["daily_dd_used"] = daily_dd
        result["daily_dd_remaining"] = max(0.0, self.config.daily_dd_limit - daily_dd)


        effective_daily_limit = self.config.daily_dd_limit - self.config.daily_dd_safety_buffer
        effective_max_limit = self.config.max_dd_limit - self.config.max_dd_safety_buffer


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


        if total_dd >= self.config.emergency_close_all_pct:

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


        return max(0.1, min(1.0, headroom))


    def calculate_lots(
        self,
        symbol: str,
        signal_strength: float = 1.0,
        stop_loss_pips: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        override_balance: Optional[float] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        details: Dict[str, Any] = {
            "symbol": symbol,
            "method": "risk_based",
            "adjustments": [],
            "prop_firm_mode": self.config.prop_firm_mode,
        }


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


        balance = self.get_current_balance() if override_balance is None else float(override_balance)
        details["balance"] = balance


        risk_pct_map = {
            RiskLevel.ULTRA_CONSERVATIVE: 0.0015,
            RiskLevel.CONSERVATIVE: 0.0025,
            RiskLevel.MODERATE: 0.005,
            RiskLevel.AGGRESSIVE: 0.0075,
            RiskLevel.VERY_AGGRESSIVE: 0.01,
        }
        base_risk_pct = risk_pct_map.get(risk_level, self.config.risk_per_trade_pct)

        risk_pct = min(base_risk_pct, self.config.max_risk_per_trade_pct)
        details["risk_pct"] = risk_pct
        details["risk_level"] = risk_level.value


        risk_amount = balance * risk_pct
        details["risk_amount"] = risk_amount


        pip_value = self.get_pip_value(symbol)
        pip_size = self.get_pip_size(symbol)
        details["pip_value_per_lot"] = pip_value
        details["pip_size"] = pip_size


        symbol_upper = self._normalize_symbol(symbol)

        if stop_loss_pips is None:
            if volatility is not None and volatility > 0:

                vol_baseline_price = self.get_volatility_baseline(symbol)
                current_price_range = float(volatility)
                baseline_price_range = float(vol_baseline_price) if vol_baseline_price > 0 else current_price_range


                stop_loss_price = max(current_price_range * 1.5, baseline_price_range * 0.5)
            else:

                if "XAU" in symbol_upper:

                    stop_loss_price = 10.0
                elif "XAG" in symbol_upper:
                    stop_loss_price = 0.30
                else:

                    stop_loss_price = 0.0100


            stop_loss_pips = max(1.0, stop_loss_price / pip_size)
        else:
            stop_loss_pips = max(1.0, float(stop_loss_pips))

        details["stop_loss_pips"] = stop_loss_pips


        if stop_loss_pips > 0 and pip_value > 0:
            base_lots = risk_amount / (stop_loss_pips * pip_value)
        else:

            contract_size = self.get_contract_size(symbol)
            leverage = self.get_current_leverage()
            max_margin_lots = (balance * leverage) / contract_size
            base_lots = max_margin_lots * risk_pct * 10.0
        details["base_lots"] = base_lots


        if self.config.signal_strength_scaling:
            sig = max(0.1, min(1.0, float(signal_strength)))
            signal_multiplier = 0.5 + (sig * 0.5)
            base_lots *= signal_multiplier
            details["adjustments"].append(f"signal_strength={sig:.2f}→{signal_multiplier:.2f}x")


        if self.config.volatility_scaling and volatility is not None and volatility > 0:
            vol_baseline_price = self.get_volatility_baseline(symbol)
            if vol_baseline_price > 0:
                vol_ratio = float(volatility) / float(vol_baseline_price)

                vol_multiplier = max(0.5, min(1.5, 1.0 / max(vol_ratio, 1e-6)))
                base_lots *= vol_multiplier
                details["adjustments"].append(f"volatility={volatility:.4f}→{vol_multiplier:.2f}x")


        if self.config.drawdown_scaling:
            drawdown = self.get_current_drawdown()
            if drawdown > 0.02:

                dd_multiplier = max(0.3, 1.0 - (drawdown * 5.0))
                base_lots *= dd_multiplier
                details["adjustments"].append(f"drawdown={drawdown:.2%}→{dd_multiplier:.2f}x")


        risk_scale = self.get_risk_scale()
        if risk_scale != 1.0:
            base_lots *= risk_scale
            details["adjustments"].append(f"risk_controller_scale={risk_scale:.2f}x")
            details["risk_scale"] = risk_scale


        mode_multiplier, mode_name = self.get_trading_mode_multiplier()
        if mode_multiplier != 1.0:
            base_lots *= mode_multiplier
            details["adjustments"].append(f"trading_mode={mode_name}→{mode_multiplier:.2f}x")
        details["trading_mode"] = mode_name
        details["trading_mode_multiplier"] = mode_multiplier


        prime_multiplier, in_prime = self.get_prime_hours_multiplier()
        if in_prime and prime_multiplier > 1.0:
            base_lots *= prime_multiplier
            details["adjustments"].append(f"prime_hours_boost→{prime_multiplier:.2f}x")
        details["in_prime_window"] = in_prime
        details["prime_hours_multiplier"] = prime_multiplier


        market_ctx_multiplier, market_ctx_details = self.get_market_context_adjustment()
        if market_ctx_multiplier < 1.0:
            base_lots *= market_ctx_multiplier
            details["adjustments"].append(f"market_context={market_ctx_multiplier:.2f}x")
        details["market_context"] = market_ctx_details
        details["market_context_multiplier"] = market_ctx_multiplier


        if self.config.prop_firm_mode:
            prop_reduction = self.get_prop_firm_lot_reduction()
            if prop_reduction == 0.0:

                details["adjustments"].append("prop_firm_BLOCKED")
                details["prop_firm_reduction"] = 0.0
                details["final_lots"] = 0.0
                details["blocked_reason"] = "prop_firm_limits_breached"
                return 0.0, details
            elif prop_reduction < 1.0:
                base_lots *= prop_reduction
                details["adjustments"].append(f"prop_firm_headroom={prop_reduction:.2f}x")
                details["prop_firm_reduction"] = prop_reduction


        lots = self._round_to_step(base_lots, self.config.lot_step)


        lots = max(self.config.min_lot, min(lots, self.config.max_lot))


        contract_size = self.get_contract_size(symbol)
        leverage = self.get_current_leverage()
        max_margin = balance * self.config.max_exposure_pct
        margin_required = (lots * contract_size) / leverage

        if margin_required > max_margin:

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
        if balance is None:
            balance = self.get_current_balance()

        contract_size = self.get_contract_size(symbol)
        leverage = self.get_current_leverage()

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
        if step <= 0:
            return value

        if mode == "down":
            return math.floor(value / step) * step
        elif mode == "up":
            return math.ceil(value / step) * step
        else:
            return round(value / step) * step

    def publish_lot_config_to_bus(self) -> None:
        if not self.bus:
            return

        try:
            config_data = {
                "account_balance": self.get_current_balance(),
                "account_leverage": self.get_current_leverage(),
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


def calculate_lots(
    symbol: str,
    signal_strength: float = 1.0,
    stop_loss_pips: Optional[float] = None,
    volatility: Optional[float] = None,
    risk_level: RiskLevel = RiskLevel.MODERATE,
) -> float:
    calculator = UnifiedLotCalculator.get_instance()
    lots, _ = calculator.calculate_lots(
        symbol=symbol,
        signal_strength=signal_strength,
        stop_loss_pips=stop_loss_pips,
        volatility=volatility,
        risk_level=risk_level,
    )
    return lots
