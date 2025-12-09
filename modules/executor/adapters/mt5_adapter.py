# modules/executor/adapters/mt5_adapter.py
from __future__ import annotations

import time
import math
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from .base_adapter import BaseLiveAdapter, LiveAdapterConfig
from modules.utils.audit_utils import RotatingLogger

# Treat MetaTrader5 as `Any` so Pylance/pyright doesn't complain about attrs.
try:
    import MetaTrader5 as _mt5_mod  # type: ignore
    mt5 = cast(Any, _mt5_mod)
    _MT5 = True
except Exception:
    mt5 = cast(Any, None)
    _MT5 = False


def _sf(v: Any, default: float = 0.0) -> float:
    """Safe float with finite check."""
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _load_sl_tp_config() -> Dict[str, Any]:
    """Load SL/TP configuration from risk_policy.yaml."""
    try:
        config_path = Path("config/risk_policy.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
                return config.get("sl_tp_settings", {})
    except Exception as e:
        print(f"[MT5Adapter] Warning: Failed to load SL/TP config: {e}")
    return {}


def _get_sl_tp_pips(symbol: str) -> Tuple[float, float]:
    """
    Get SL/TP pips for a symbol from config.

    NOTE: These are EMERGENCY safety nets only.
    Real exit logic is handled by SmartPositionManager based on voting signals.
    """
    config = _load_sl_tp_config()

    # SL
    if not config.get("auto_sl_enabled", True):
        sl_pips = 0
    else:
        symbol_config = config.get(symbol, config.get("default", {}))
        sl_pips = symbol_config.get("stop_loss_pips", 200)  # wide emergency SL

    # TP (default off — SmartPositionManager handles exits)
    if not config.get("auto_tp_enabled", False):
        tp_pips = 0
    else:
        symbol_config = config.get(symbol, config.get("default", {}))
        tp_pips = symbol_config.get("take_profit_pips", 0)

    return float(sl_pips), float(tp_pips)


def _pips_to_price(symbol: str, pips: float) -> float:
    """Convert pips to price distance based on symbol."""
    sym_upper = symbol.upper()
    # Gold (XAU) uses 0.01 per pip; Forex typically 0.0001 (or 0.01 for JPY pairs)
    if "XAU" in sym_upper or "GOLD" in sym_upper:
        return pips * 0.01  # 1 pip = $0.01 for gold
    if "XAG" in sym_upper or "SILVER" in sym_upper:
        return pips * 0.01  # treat similarly for simplicity
    if "JPY" in sym_upper:
        return pips * 0.01  # 1 pip = 0.01 for JPY pairs
    return pips * 0.0001  # default: 1 pip = 0.0001


class MT5Adapter(BaseLiveAdapter):
    """
    MT5 adapter wired to BaseLiveAdapter's robust wrappers.
    Only *_impl methods below talk to MT5 directly.
    """

    # Optional explicit overrides (units per 1.0 lot)
    SYMBOL_CONTRACT_SIZES: Dict[str, float] = {
        "XAUUSD": 100.0,   # 100 oz per lot
        "XAUEUR": 100.0,
        "XAGUSD": 5000.0,  # 5000 oz per lot for silver
        "XAGEUR": 5000.0,
        "BTCUSD": 1.0,     # 1 BTC per lot
        "ETHUSD": 1.0,     # 1 ETH per lot
        # Forex pairs fall back to default FX contract size in BaseLiveAdapter
    }

    def __init__(self, cfg: LiveAdapterConfig):
        super().__init__(cfg)
        self._working_fill_mode: Dict[str, int] = {}  # Cache working fill modes per symbol
        try:
            self.log = RotatingLogger(
                "MT5Adapter",
                log_path="logs/executor/mt5_adapter.log",
                operator_mode=True,
            )
        except Exception:
            class _Dummy:
                def info(self, *a, **k):
                    pass

                def warning(self, *a, **k):
                    pass

                def error(self, *a, **k):
                    pass

                def debug(self, *a, **k):
                    pass

            self.log = _Dummy()

    # ─────────────────────────────────────────────────────
    # Contract sizing override (aligned with Executor)
    # ─────────────────────────────────────────────────────
    def contract_size_for(self, instrument: str) -> float:
        """
        Symbol-specific contract size override.

        Keeps MT5 adapter aligned with Executor._get_contract_size and
        BaseLiveAdapter defaults, but allows explicit per-symbol overrides.
        """
        sym_upper = (instrument or "").upper().replace("_", "").replace("/", "")
        if sym_upper in self.SYMBOL_CONTRACT_SIZES:
            return float(self.SYMBOL_CONTRACT_SIZES[sym_upper])
        # Fall back to base logic (XAU/XAG/BTC/ETH/FX)
        return super().contract_size_for(instrument)

    def _get_contract_size(self, symbol: str, default: float = 100_000.0) -> float:
        """
        Backwards-compatible helper used internally in this adapter.

        Delegates to contract_size_for() and falls back to provided default.
        """
        cs = self.contract_size_for(symbol)
        if cs > 0:
            return cs
        return float(default)

    # ─────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────
    def _connect_impl(self) -> bool:
        if not _MT5:
            return False
        try:
            # Load credentials from environment or config
            from live.mt5_credentials import MT5Credentials

            max_retries = 5
            retry_delay = 3.0

            for attempt in range(max_retries):
                try:
                    # Ensure clean state
                    try:
                        mt5.shutdown()
                    except Exception:
                        pass

                    ok = mt5.initialize(
                        login=MT5Credentials.ACCOUNT,
                        password=MT5Credentials.PASSWORD,
                        server=MT5Credentials.SERVER,
                        timeout=60000,
                    )
                    if ok:
                        account_info = mt5.account_info()
                        if account_info:
                            try:
                                self.log.info(
                                    f"[MT5] Connected: "
                                    f"login={getattr(account_info, 'login', None)} "
                                    f"balance={getattr(account_info, 'balance', 0.0):.2f}"
                                )
                            except Exception:
                                pass
                            return True

                    # Failed, optional retry
                    if attempt < max_retries - 1:
                        try:
                            self.log.warning(
                                f"[MT5] initialize/login failed, retrying "
                                f"({attempt + 1}/{max_retries})"
                            )
                        except Exception:
                            pass
                        time.sleep(retry_delay * (attempt + 1))

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        try:
                            self.log.error(f"[MT5] Connect error: {e}")
                        except Exception:
                            pass
                        raise

            return False
        except Exception:
            return False

    def _disconnect_impl(self) -> None:
        if not _MT5:
            return
        try:
            mt5.shutdown()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    # Account & market
    # ─────────────────────────────────────────────────────
    def _get_account_info_impl(self) -> Dict[str, float | str]:
        if not (_MT5 and self.connected):
            return {}
        try:
            ai = mt5.account_info()
            if not ai:
                return {}
            return {
                "balance": _sf(getattr(ai, "balance", 0.0)),
                "equity": _sf(getattr(ai, "equity", 0.0)),
                "margin": _sf(getattr(ai, "margin", 0.0)),
                "free_margin": _sf(getattr(ai, "margin_free", 0.0)),
                "margin_level": _sf(getattr(ai, "margin_level", 0.0)),
                "leverage": _sf(getattr(ai, "leverage", 100.0)),  # Fetch from MT5
                "currency": str(getattr(ai, "currency", "EUR")),
            }
        except Exception:
            return {}

    def _ensure_symbol(self, symbol: str) -> bool:
        if not _MT5:
            return False
        try:
            info = mt5.symbol_info(symbol)
            if info and getattr(info, "visible", False):
                return True
            ok = bool(mt5.symbol_select(symbol, True))
            if not ok:
                try:
                    last_err = (
                        mt5.last_error() if hasattr(mt5, "last_error") else (None, None)
                    )
                    self.log.warning(
                        f"[MT5] symbol_select failed: {symbol}; last_error={last_err}"
                    )
                except Exception:
                    pass
            return ok
        except Exception:
            return False

    def _get_prices_impl(self, instrument: str) -> Dict[str, float]:
        """
        Return floats only: {'bid': float, 'ask': float, 'mid': float, 'ts': float}.
        On error, returns a zeroed snapshot with current timestamp.
        """
        now_ts = float(time.time())
        if not (_MT5 and self.connected):
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": now_ts}
        try:
            if not self._ensure_symbol(instrument):
                try:
                    self.log.warning(
                        f"[MT5] No prices: symbol not available: {instrument}"
                    )
                except Exception:
                    pass
                return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": now_ts}

            tick = mt5.symbol_info_tick(instrument)
            if not tick:
                try:
                    self.log.warning(
                        f"[MT5] symbol_info_tick returned None for {instrument}"
                    )
                except Exception:
                    pass
                return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": now_ts}

            bid = _sf(getattr(tick, "bid", 0.0))
            ask = _sf(getattr(tick, "ask", 0.0))
            if bid > 0.0 and ask > 0.0:
                mid = 0.5 * (bid + ask)
            elif bid > 0.0:
                mid = bid
            elif ask > 0.0:
                mid = ask
            else:
                mid = 0.0
            ts = _sf(getattr(tick, "time", now_ts), now_ts)
            return {"bid": bid, "ask": ask, "mid": mid, "ts": ts}
        except Exception:
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": now_ts}

    # ─────────────────────────────────────────────────────
    # Execution helpers
    # ─────────────────────────────────────────────────────
    def _pick_filling_mode(self, sym: str) -> int:
        """Pick a reasonable fill mode; fallback to IOC."""
        try:
            info = mt5.symbol_info(sym)
            fm = int(getattr(info, "filling_mode", -1))
            if fm in (
                getattr(mt5, "ORDER_FILLING_IOC", 1),
                getattr(mt5, "ORDER_FILLING_FOK", 0),
            ):
                return fm
        except Exception:
            pass
        return getattr(mt5, "ORDER_FILLING_IOC", 1)

    def _send_deal(
        self,
        sym: str,
        side: int,
        lots: float,
        filling_mode: Optional[int] = None,
        sl_price: Optional[float] = None,
        tp_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        if not self._ensure_symbol(sym):
            try:
                self.log.warning(
                    f"[MT5] send_deal blocked: symbol not available: {sym}"
                )
            except Exception:
                pass
            return {"ok": False, "error": "symbol_not_available"}
        try:
            order_type = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL

            # Get current price for SL/TP calculation
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                self.log.warning(f"[MT5] Could not get tick for {sym}")
                return {"ok": False, "error": "no_tick_data"}

            current_price = tick.ask if side > 0 else tick.bid

            # Calculate SL/TP if not provided
            if sl_price is None or tp_price is None:
                sl_pips, tp_pips = _get_sl_tp_pips(sym)
                sl_distance = _pips_to_price(sym, sl_pips)
                tp_distance = _pips_to_price(sym, tp_pips)

                if side > 0:  # BUY
                    if sl_price is None and sl_pips > 0:
                        sl_price = current_price - sl_distance
                    if tp_price is None and tp_pips > 0:
                        tp_price = current_price + tp_distance
                else:  # SELL
                    if sl_price is None and sl_pips > 0:
                        sl_price = current_price + sl_distance
                    if tp_price is None and tp_pips > 0:
                        tp_price = current_price - tp_distance

            # Round prices to symbol's digits
            symbol_info = mt5.symbol_info(sym)
            digits = getattr(symbol_info, "digits", 5) if symbol_info else 5

            if sl_price:
                sl_price = round(sl_price, digits)
            if tp_price:
                tp_price = round(tp_price, digits)

            req: Dict[str, Any] = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": float(lots),
                "type": order_type,
                "deviation": 20,  # points
                "magic": 424242,
                "comment": "executor",
                "type_filling": (
                    filling_mode
                    if filling_mode is not None
                    else self._pick_filling_mode(sym)
                ),
                "type_time": mt5.ORDER_TIME_GTC,
            }

            # Add SL/TP to request if valid
            if sl_price and sl_price > 0:
                req["sl"] = sl_price
            if tp_price and tp_price > 0:
                req["tp"] = tp_price

            try:
                sl_str = f"SL={sl_price:.5f}" if sl_price else "SL=None"
                tp_str = f"TP={tp_price:.5f}" if tp_price else "TP=None"
                self.log.info(
                    f"[MT5] order_send: sym={sym} side={side} lots={lots:.4f} "
                    f"{sl_str} {tp_str} filling={req['type_filling']}"
                )
            except Exception:
                pass

            r = mt5.order_send(req)
            if r is None:
                try:
                    last_err = (
                        mt5.last_error() if hasattr(mt5, "last_error") else (None, None)
                    )
                    self.log.error(
                        f"[MT5] order_send returned None; last_error={last_err}"
                    )
                except Exception:
                    pass
                return {"ok": False, "error": "order_send_none"}

            ret_ok = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
            ret_placed = getattr(mt5, "TRADE_RETCODE_PLACED", 10008)
            ret_partial = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)
            ret_invalid_fill = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)

            if r.retcode in (ret_ok, ret_placed, ret_partial):
                ticket = getattr(r, 'order', getattr(r, 'deal', 0))
                try:
                    # Log successful order WITH SL/TP confirmation
                    sl_status = f"SL={sl_price:.5f}" if sl_price else "SL=NONE⚠️"
                    tp_status = f"TP={tp_price:.5f}" if tp_price else "TP=NONE"
                    self.log.info(
                        f"[MT5] ✅ ORDER SUCCESS: {sym} {'BUY' if side > 0 else 'SELL'} {lots:.4f} lots | "
                        f"ticket={ticket} price={_sf(getattr(r, 'price', 0.0)):.5f} | "
                        f"{sl_status} {tp_status}"
                    )
                    # CRITICAL: Warn if SL was not set
                    if not sl_price:
                        self.log.warning(
                            f"[MT5] ⚠️ WARNING: Order {ticket} has NO STOP-LOSS! "
                            "Position is unprotected!"
                        )
                except Exception:
                    pass
                return {
                    "ok": True,
                    "price": _sf(getattr(r, "price", 0.0)),
                    "lots": float(lots),
                    "side": 1 if side > 0 else -1,
                    "ticket": ticket,
                    "sl": sl_price,
                    "tp": tp_price,
                }

            # One retry with alternate filling mode if fill mode invalid
            if r.retcode in (
                ret_invalid_fill,
                getattr(mt5, "TRADE_RETCODE_INVALID", 10006),
            ):
                alt = (
                    getattr(mt5, "ORDER_FILLING_FOK", 0)
                    if req["type_filling"] == getattr(mt5, "ORDER_FILLING_IOC", 1)
                    else getattr(mt5, "ORDER_FILLING_IOC", 1)
                )
                r2 = mt5.order_send({**req, "type_filling": alt})
                if r2 and r2.retcode in (ret_ok, ret_placed, ret_partial):
                    try:
                        self.log.info(
                            f"[MT5] order_send retry OK: retcode={r2.retcode} "
                            f"price={_sf(getattr(r2, 'price', 0.0)):.5f} "
                            f"ticket={getattr(r2, 'order', getattr(r2, 'deal', 0))}"
                        )
                    except Exception:
                        pass
                    return {
                        "ok": True,
                        "price": _sf(getattr(r2, "price", 0.0)),
                        "lots": float(lots),
                        "side": 1 if side > 0 else -1,
                        "ticket": getattr(r2, "order", getattr(r2, "deal", 0)),
                    }

            try:
                last_err = (
                    mt5.last_error() if hasattr(mt5, "last_error") else (None, None)
                )
                self.log.error(
                    f"[MT5] order_send failed: retcode={getattr(r, 'retcode', 'unknown')} "
                    f"last_error={last_err}"
                )
            except Exception:
                pass
            return {"ok": False, "error": f"retcode={getattr(r, 'retcode', 'unknown')}"}
        except Exception as e:
            try:
                self.log.error(f"[MT5] order_send exception: {e}")
            except Exception:
                pass
            return {"ok": False, "error": str(e)}

    # ─────────────────────────────────────────────────────
    # Execution impls used by BaseLiveAdapter wrappers
    # ─────────────────────────────────────────────────────
    def _market_order_impl(
        self,
        instrument: str,
        side: int,
        lots: float,
    ) -> Dict[str, Any]:
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}
        # side follows BaseLiveAdapter semantics: +1 = BUY, -1 = SELL
        return self._send_deal(instrument, side, lots)

    def _reduce_position_impl(
        self,
        instrument: str,
        lots: float,
        side: int,
    ) -> Dict[str, Any]:
        """
        Reduce (partially close) a position by sending a market order.

        The `side` argument is interpreted exactly like _market_order_impl:
        +1 => BUY, -1 => SELL. The caller is responsible for choosing the
        correct direction (typically opposite to the net position).
        """
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}
        return self._send_deal(instrument, side, lots)

    def _close_position_impl(self, instrument: str) -> Dict[str, Any]:
        """
        Close all positions for instrument BY TICKET (required for hedging accounts).
        On hedging accounts, sending opposite-side orders creates hedges instead of closing.
        We MUST specify the position ticket to properly close.
        """
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}
        try:
            if not self._ensure_symbol(instrument):
                return {"ok": False, "error": "symbol_not_available"}

            pos_list = list(mt5.positions_get(symbol=instrument) or [])
            if not pos_list:
                return {"ok": True, "price": 0.0, "ticket": 0}

            last_px = 0.0
            for p in pos_list:
                ticket = getattr(p, "ticket", 0)
                pos_type = getattr(p, "type", 1)
                lots = _sf(getattr(p, "volume", 0.0))
                if lots <= 0.0 or ticket == 0:
                    continue

                # Determine close order type (opposite of position)
                close_type = (
                    mt5.ORDER_TYPE_SELL
                    if pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0)
                    else mt5.ORDER_TYPE_BUY
                )

                # Get current price
                tick = mt5.symbol_info_tick(instrument)
                if tick is None:
                    self.log.warning(f"[MT5] close_position: No tick for {instrument}")
                    return {"ok": False, "error": "no_tick_data"}

                price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

                # Use cached filling mode first, then try others
                all_modes = [
                    getattr(mt5, "ORDER_FILLING_IOC", 1),
                    getattr(mt5, "ORDER_FILLING_FOK", 0),
                    getattr(mt5, "ORDER_FILLING_RETURN", 2),
                ]
                cached_mode = self._working_fill_mode.get(instrument)
                if cached_mode is not None:
                    filling_modes = [cached_mode] + [m for m in all_modes if m != cached_mode]
                else:
                    filling_modes = all_modes

                ret_ok = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
                ret_no_prices = 10021  # Market closed / no quotes
                ret_market_closed = 10018
                ret_invalid_fill = 10030  # Unsupported filling mode

                success = False
                last_error = None

                for fill_mode in filling_modes:
                    # Build close request WITH position ticket (required for hedging accounts)
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": instrument,
                        "volume": float(lots),
                        "type": close_type,
                        "position": ticket,  # CRITICAL: specify ticket for hedging accounts
                        "price": price,
                        "deviation": 20,
                        "magic": 424242,
                        "comment": "close_position",
                        "type_filling": fill_mode,
                        "type_time": mt5.ORDER_TIME_GTC,
                    }

                    self.log.info(
                        f"[MT5] close_position: ticket={ticket} sym={instrument} "
                        f"lots={lots:.4f} type={'SELL' if close_type == mt5.ORDER_TYPE_SELL else 'BUY'} "
                        f"fill_mode={fill_mode}"
                    )

                    r = mt5.order_send(request)
                    if r is None:
                        last_error = "order_send_none"
                        continue

                    if r.retcode == ret_ok:
                        last_px = _sf(getattr(r, "price", 0.0), last_px)
                        self.log.info(f"[MT5] close_position: ✅ Closed ticket {ticket} @ {last_px:.5f}")
                        self._working_fill_mode[instrument] = fill_mode  # Cache working mode
                        success = True
                        break
                    elif r.retcode == ret_invalid_fill:
                        # Try next filling mode
                        self.log.debug(f"[MT5] close_position: fill_mode={fill_mode} not supported, trying next")
                        last_error = f"retcode_{r.retcode}"
                        continue
                    elif r.retcode in (ret_no_prices, ret_market_closed):
                        # Market closed - don't spam errors
                        self.log.warning(
                            f"[MT5] close_position: Market closed for {instrument} "
                            f"(retcode={r.retcode}). Will retry when market opens."
                        )
                        return {"ok": False, "error": "market_closed", "retcode": r.retcode}
                    else:
                        last_error = f"retcode_{r.retcode}: {getattr(r, 'comment', '')}"
                        self.log.error(f"[MT5] close_position: ❌ Failed ticket {ticket}: {last_error}")
                        break  # Don't try other fill modes for non-fill errors

                if not success:
                    self.log.error(f"[MT5] close_position: All filling modes failed for ticket {ticket}")
                    return {"ok": False, "error": last_error or "all_fills_failed"}

            return {"ok": True, "price": last_px}
        except Exception as e:
            self.log.error(f"[MT5] close_position exception: {e}")
            return {"ok": False, "error": str(e)}

    def _modify_position_impl(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Concrete implementation for BaseLiveAdapter.modify_position wrapper.

        Args:
            ticket: Position ticket to modify
            sl: New stop loss price (None = don't change)
            tp: New take profit price (None = don't change)
        """
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}

        try:
            # Get current position
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return {"ok": False, "error": f"position_not_found: {ticket}"}

            pos = positions[0]
            symbol = getattr(pos, "symbol", "")
            current_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            current_tp = float(getattr(pos, "tp", 0.0) or 0.0)

            # Use current values if not changing
            new_sl = sl if sl is not None else current_sl
            new_tp = tp if tp is not None else current_tp

            # Skip if no change needed
            if abs(new_sl - current_sl) < 0.00001 and abs(new_tp - current_tp) < 0.00001:
                return {"ok": True, "message": "no_change_needed", "sl": current_sl, "tp": current_tp}

            # Ensure symbol is selected
            if not self._ensure_symbol(symbol):
                return {"ok": False, "error": "symbol_not_available"}

            # Build modify request
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": ticket,
                "sl": new_sl,
                "tp": new_tp,
            }

            self.log.debug(
                f"[MT5] modify_position: ticket={ticket} sl={current_sl:.5f}->{new_sl:.5f} "
                f"tp={current_tp:.5f}->{new_tp:.5f}"
            )

            result = mt5.order_send(request)

            if result is None:
                error = mt5.last_error()
                self.log.error(f"[MT5] modify_position: ❌ Failed: {error}")
                return {"ok": False, "error": str(error)}

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                self.log.info(
                    f"[MT5] modify_position: ✅ Modified ticket {ticket} "
                    f"SL={new_sl:.5f} TP={new_tp:.5f}"
                )
                return {"ok": True, "sl": new_sl, "tp": new_tp}
            else:
                error_msg = f"retcode={result.retcode}: {result.comment}"
                self.log.error(f"[MT5] modify_position: ❌ {error_msg}")
                return {"ok": False, "error": error_msg}

        except Exception as e:
            self.log.error(f"[MT5] modify_position exception: {e}")
            return {"ok": False, "error": str(e)}

    def _sync_positions_impl(self) -> Dict[str, Dict[str, Any]]:
        """
        Net positions per symbol:

            side = sign(buy_lots - sell_lots)
            units = abs(net_lots) * contract_size_for(symbol)
            entry_price = VWAP of net side
        """
        if not (_MT5 and self.connected):
            return {}

        out: Dict[str, Dict[str, Any]] = {}
        try:
            ps = list(mt5.positions_get() or [])
            if not ps:
                return {}
            try:
                self.log.info(f"[MT5] sync_positions: {len(ps)} raw positions")
            except Exception:
                pass

            # group by symbol
            by_sym: Dict[str, List[Any]] = {}
            for p in ps:
                sym = str(getattr(p, "symbol", "") or "")
                if not sym:
                    continue
                by_sym.setdefault(sym, []).append(p)

            for sym, plist in by_sym.items():
                # Use unified contract size logic
                cs = self.contract_size_for(sym)

                buy_lots = sum(
                    _sf(getattr(p, "volume", 0.0))
                    for p in plist
                    if getattr(p, "type", 1)
                    == getattr(mt5, "POSITION_TYPE_BUY", 0)
                )
                sell_lots = sum(
                    _sf(getattr(p, "volume", 0.0))
                    for p in plist
                    if getattr(p, "type", 1)
                    == getattr(mt5, "POSITION_TYPE_SELL", 1)
                )
                net_lots = buy_lots - sell_lots
                if abs(net_lots) <= 0.0:
                    continue

                side = 1 if net_lots > 0 else -1
                units = abs(net_lots) * cs

                def vwavg(pl: List[Any], pos_type: int) -> float:
                    vols: List[float] = []
                    prices: List[float] = []
                    for q in pl:
                        if getattr(q, "type", 1) == pos_type:
                            v = _sf(getattr(q, "volume", 0.0))
                            px = _sf(getattr(q, "price_open", 0.0))
                            if v > 0 and px > 0:
                                vols.append(v)
                                prices.append(px)
                    tot = sum(vols)
                    return (
                        float(
                            sum(v * px for v, px in zip(vols, prices)) / tot
                        )
                        if tot > 0
                        else 0.0
                    )

                entry_price = (
                    vwavg(plist, getattr(mt5, "POSITION_TYPE_BUY", 0))
                    if side > 0
                    else vwavg(plist, getattr(mt5, "POSITION_TYPE_SELL", 1))
                )
                notional_eur = units * entry_price

                times: List[float] = []
                total_profit = 0.0
                current_price = 0.0
                primary_ticket = 0
                primary_lots = 0.0
                position_sl = 0.0
                position_tp = 0.0
                
                for q in plist:
                    t = getattr(q, "time", None)
                    if isinstance(t, (int, float)) and math.isfinite(t):
                        times.append(float(t))
                    
                    # Sum up profit from all positions for this symbol
                    profit = _sf(getattr(q, "profit", 0.0))
                    total_profit += profit
                    
                    # Get current price from any position
                    if current_price == 0.0:
                        current_price = _sf(getattr(q, "price_current", 0.0))
                    
                    # Track the largest ticket for SL/TP modifications
                    vol = _sf(getattr(q, "volume", 0.0))
                    if vol > primary_lots:
                        primary_lots = vol
                        primary_ticket = int(getattr(q, "ticket", 0) or 0)
                        position_sl = _sf(getattr(q, "sl", 0.0))
                        position_tp = _sf(getattr(q, "tp", 0.0))
                
                open_time = min(times) if times else 0.0

                out[sym] = {
                    "instrument": sym,
                    "side": side,
                    "units": float(units),
                    "entry_price": float(entry_price),
                    "notional_eur": float(notional_eur),
                    "open_time": open_time,
                    # NEW: Add P&L and price data for experts/PPO
                    "unrealized_pnl": float(total_profit),
                    "profit": float(total_profit),  # alias for compatibility
                    "current_price": float(current_price),
                    "price_current": float(current_price),  # alias
                    "ticket": primary_ticket,
                    "sl": float(position_sl),
                    "tp": float(position_tp),
                    "lots": float(abs(net_lots)),  # for convenience
                }

            return out
        except Exception:
            return {}

    # ─────────────────────────────────────────────────────
    # SL/TP fixer (optional hard safety belt)
    # ─────────────────────────────────────────────────────
    def fix_positions_without_sl_tp(self) -> Dict[str, Any]:
        """
        Check all open positions and add SL/TP if missing,
        based on risk_policy.yaml sl_tp_settings.

        This is a last-resort safety belt in case of disconnects,
        not the primary exit logic.
        """
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected", "fixed": 0}

        try:
            config = _load_sl_tp_config()
            if not config.get("fix_missing_sl_tp", True):
                return {"ok": True, "message": "fix_missing_sl_tp disabled", "fixed": 0}

            positions = list(mt5.positions_get() or [])
            if not positions:
                return {"ok": True, "message": "no_positions", "fixed": 0}

            fixed_count = 0
            results: List[Dict[str, Any]] = []

            for pos in positions:
                ticket = getattr(pos, "ticket", 0)
                symbol = getattr(pos, "symbol", "")
                current_sl = _sf(getattr(pos, "sl", 0.0))
                current_tp = _sf(getattr(pos, "tp", 0.0))
                open_price = _sf(getattr(pos, "price_open", 0.0))
                pos_type = getattr(pos, "type", 0)

                # Check if SL or TP is missing
                needs_sl = current_sl <= 0 and config.get("auto_sl_enabled", True)
                needs_tp = current_tp <= 0 and config.get("auto_tp_enabled", True)

                if not needs_sl and not needs_tp:
                    continue

                # Get SL/TP pips for this symbol
                sl_pips, tp_pips = _get_sl_tp_pips(symbol)
                sl_distance = _pips_to_price(symbol, sl_pips)
                tp_distance = _pips_to_price(symbol, tp_pips)

                # Get symbol info for rounding
                symbol_info = mt5.symbol_info(symbol)
                digits = getattr(symbol_info, "digits", 5) if symbol_info else 5

                new_sl = current_sl
                new_tp = current_tp
                is_buy = pos_type == getattr(mt5, "POSITION_TYPE_BUY", 0)

                if is_buy:
                    if needs_sl and sl_pips > 0:
                        new_sl = round(open_price - sl_distance, digits)
                    if needs_tp and tp_pips > 0:
                        new_tp = round(open_price + tp_distance, digits)
                else:  # SELL
                    if needs_sl and sl_pips > 0:
                        new_sl = round(open_price + sl_distance, digits)
                    if needs_tp and tp_pips > 0:
                        new_tp = round(open_price - tp_distance, digits)

                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_sl if new_sl > 0 else 0.0,
                    "tp": new_tp if new_tp > 0 else 0.0,
                }

                try:
                    self.log.info(
                        f"[MT5] Fixing SL/TP for position {ticket}: {symbol} "
                        f"SL={new_sl:.5f} TP={new_tp:.5f}"
                    )
                except Exception:
                    pass

                result = mt5.order_send(request)

                if result and result.retcode in (
                    getattr(mt5, "TRADE_RETCODE_DONE", 10009),
                ):
                    fixed_count += 1
                    results.append(
                        {
                            "ticket": ticket,
                            "symbol": symbol,
                            "sl": new_sl,
                            "tp": new_tp,
                            "ok": True,
                        }
                    )
                    try:
                        self.log.info(
                            f"[MT5] Fixed position {ticket}: "
                            f"SL={new_sl:.5f} TP={new_tp:.5f}"
                        )
                    except Exception:
                        pass
                else:
                    error = getattr(result, "retcode", "unknown") if result else "none"
                    results.append(
                        {"ticket": ticket, "symbol": symbol, "ok": False, "error": str(error)}
                    )
                    try:
                        self.log.warning(
                            f"[MT5] Failed to fix position {ticket}: retcode={error}"
                        )
                    except Exception:
                        pass

            return {
                "ok": True,
                "fixed": fixed_count,
                "total": len(positions),
                "results": results,
            }

        except Exception as e:
            try:
                self.log.error(f"[MT5] fix_positions_without_sl_tp exception: {e}")
            except Exception:
                pass
            return {"ok": False, "error": str(e), "fixed": 0}
