# modules/executor/adapters/mt5_adapter.py
from __future__ import annotations

import time
import math
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

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
    """Load SL/TP configuration from risk_policy.yaml"""
    try:
        config_path = Path("config/risk_policy.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                return config.get('sl_tp_settings', {})
    except Exception:
        pass
    return {}


def _get_sl_tp_pips(symbol: str) -> tuple:
    """Get SL/TP pips for a symbol from config"""
    config = _load_sl_tp_config()
    
    # Check if auto SL/TP is enabled
    if not config.get('auto_sl_enabled', True):
        sl_pips = 0
    else:
        sl_pips = config.get(symbol, config.get('default', {})).get('stop_loss_pips', 50)
    
    if not config.get('auto_tp_enabled', True):
        tp_pips = 0
    else:
        tp_pips = config.get(symbol, config.get('default', {})).get('take_profit_pips', 100)
    
    return (sl_pips, tp_pips)


def _pips_to_price(symbol: str, pips: float) -> float:
    """Convert pips to price distance based on symbol"""
    # Gold (XAU) uses 0.01 per pip, Forex typically uses 0.0001 or 0.01 for JPY pairs
    sym_upper = symbol.upper()
    if 'XAU' in sym_upper or 'GOLD' in sym_upper:
        return pips * 0.01  # 1 pip = $0.01 for gold
    elif 'JPY' in sym_upper:
        return pips * 0.01  # 1 pip = 0.01 for JPY pairs
    else:
        return pips * 0.0001  # 1 pip = 0.0001 for most forex pairs


class MT5Adapter(BaseLiveAdapter):
    """
    MT5 adapter wired to BaseLiveAdapter's robust wrappers.
    Only *_impl methods below talk to MT5 directly.
    """
    
    # Symbol-specific contract sizes (units per 1.0 lot)
    # Gold/Silver are in oz, Forex in currency units
    SYMBOL_CONTRACT_SIZES = {
        'XAUUSD': 100.0,      # 100 oz per lot
        'XAUEUR': 100.0,      # 100 oz per lot
        'XAGUSD': 5000.0,     # 5000 oz per lot for silver
        'XAGEUR': 5000.0,     # 5000 oz per lot for silver
        'BTCUSD': 1.0,        # 1 BTC per lot
        'ETHUSD': 1.0,        # 1 ETH per lot
        # Forex pairs use default (100,000)
    }

    def __init__(self, cfg: LiveAdapterConfig):
        super().__init__(cfg)
        try:
            self.log = RotatingLogger("MT5Adapter", log_path="logs/executor/mt5_adapter.log", operator_mode=True)
        except Exception:
            class _Dummy:
                def info(self, *a, **k):
                    pass
                def warning(self, *a, **k):
                    pass
                def error(self, *a, **k):
                    pass
            self.log = _Dummy()
    
    def _get_contract_size(self, symbol: str, default: float = 100_000.0) -> float:
        """Get contract size for a symbol (units per 1.0 lot)."""
        sym_upper = symbol.upper().replace("_", "")
        
        # Check exact match first
        if sym_upper in self.SYMBOL_CONTRACT_SIZES:
            return self.SYMBOL_CONTRACT_SIZES[sym_upper]
        
        # Check for gold/silver patterns
        if 'XAU' in sym_upper or 'GOLD' in sym_upper:
            return 100.0  # Gold: 100 oz per lot
        if 'XAG' in sym_upper or 'SILVER' in sym_upper:
            return 5000.0  # Silver: 5000 oz per lot
        
        # Crypto patterns
        if 'BTC' in sym_upper or 'ETH' in sym_upper:
            return 1.0
        
        # Default to forex contract size
        return default

    # ─────────────────────────────────────────────────────
    # Connection
    # ─────────────────────────────────────────────────────
    def _connect_impl(self) -> bool:
        if not _MT5:
            return False
        try:
            # Load credentials from environment or config
            from live.mt5_credentials import MT5Credentials

            # Attempt to connect with credentials and retry logic
            max_retries = 5
            retry_delay = 3.0

            for attempt in range(max_retries):
                try:
                    # Shutdown any existing connection first
                    try:
                        mt5.shutdown()
                    except:
                        pass

                    # Initialize with credentials
                    ok = mt5.initialize(
                        login=MT5Credentials.ACCOUNT,
                        password=MT5Credentials.PASSWORD,
                        server=MT5Credentials.SERVER,
                        timeout=60000
                    )

                    if ok:
                        # Verify connection
                        account_info = mt5.account_info()
                        if account_info:
                            try:
                                self.log.info(f"[MT5] Connected: login={getattr(account_info,'login',None)} balance={getattr(account_info,'balance',0.0):.2f}")
                            except Exception:
                                pass
                            return True

                    # Failed, wait before retry
                    if attempt < max_retries - 1:
                        try:
                            self.log.warning(f"[MT5] initialize/login failed, retrying ({attempt+1}/{max_retries})")
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
    def _get_account_info_impl(self) -> Dict[str, float]:
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
                    last_err = mt5.last_error() if hasattr(mt5, 'last_error') else (None, None)
                    self.log.warning(f"[MT5] symbol_select failed: {symbol}; last_error={last_err}")
                except Exception:
                    pass
            return ok
        except Exception:
            return False

    def _get_prices_impl(self, instrument: str) -> Dict[str, float]:
        """
        Return floats only: {'bid': float, 'ask': float, 'mid': float, 'ts': float}
        (No None values to keep Pylance happy.)
        """
        if not (_MT5 and self.connected):
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": float(time.time())}
        try:
            if not self._ensure_symbol(instrument):
                try:
                    self.log.warning(f"[MT5] No prices: symbol not available: {instrument}")
                except Exception:
                    pass
                return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": float(time.time())}

            tick = mt5.symbol_info_tick(instrument)
            if not tick:
                try:
                    self.log.warning(f"[MT5] symbol_info_tick returned None for {instrument}")
                except Exception:
                    pass
                return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": float(time.time())}

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
            ts = _sf(getattr(tick, "time", time.time()), time.time())
            return {"bid": bid, "ask": ask, "mid": mid, "ts": ts}
        except Exception:
            return {"bid": 0.0, "ask": 0.0, "mid": 0.0, "ts": float(time.time())}

    # ─────────────────────────────────────────────────────
    # Execution helpers
    # ─────────────────────────────────────────────────────
    def _pick_filling_mode(self, sym: str) -> int:
        """Pick a reasonable fill mode; fallback to IOC."""
        try:
            info = mt5.symbol_info(sym)
            fm = int(getattr(info, "filling_mode", -1))
            if fm in (getattr(mt5, "ORDER_FILLING_IOC", 1), getattr(mt5, "ORDER_FILLING_FOK", 0)):
                return fm
        except Exception:
            pass
        return getattr(mt5, "ORDER_FILLING_IOC", 1)

    def _send_deal(self, sym: str, side: int, lots: float, filling_mode: Optional[int] = None, 
                    sl_price: Optional[float] = None, tp_price: Optional[float] = None) -> Dict[str, Any]:
        if not self._ensure_symbol(sym):
            try:
                self.log.warning(f"[MT5] send_deal blocked: symbol not available: {sym}")
            except Exception:
                pass
            return {"ok": False, "error": "symbol_not_available"}
        try:
            t = mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL
            
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
            digits = getattr(symbol_info, 'digits', 5) if symbol_info else 5
            
            if sl_price:
                sl_price = round(sl_price, digits)
            if tp_price:
                tp_price = round(tp_price, digits)
            
            req = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": sym,
                "volume": float(lots),
                "type": t,
                "deviation": 20,  # points
                "magic": 424242,
                "comment": "executor",
                "type_filling": filling_mode if filling_mode is not None else self._pick_filling_mode(sym),
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
                self.log.info(f"[MT5] order_send: sym={sym} side={side} lots={lots:.4f} {sl_str} {tp_str} filling={req['type_filling']}")
            except Exception:
                pass
            r = mt5.order_send(req)
            if r is None:
                try:
                    last_err = mt5.last_error() if hasattr(mt5, 'last_error') else (None, None)
                    self.log.error(f"[MT5] order_send returned None; last_error={last_err}")
                except Exception:
                    pass
                return {"ok": False, "error": "order_send_none"}

            ret_ok = getattr(mt5, "TRADE_RETCODE_DONE", 10009)
            ret_placed = getattr(mt5, "TRADE_RETCODE_PLACED", 10008)
            ret_partial = getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", 10010)
            ret_invalid_fill = getattr(mt5, "TRADE_RETCODE_INVALID_FILL", 10030)

            if r.retcode in (ret_ok, ret_placed, ret_partial):
                try:
                    self.log.info(f"[MT5] order_send OK: retcode={r.retcode} price={_sf(getattr(r,'price',0.0)):.5f} ticket={getattr(r,'order',getattr(r,'deal',0))}")
                except Exception:
                    pass
                return {
                    "ok": True,
                    "price": _sf(getattr(r, "price", 0.0)),
                    "lots": float(lots),
                    "side": 1 if side > 0 else -1,
                    "ticket": getattr(r, "order", getattr(r, "deal", 0)),
                }

            # one retry with alternate filling
            if r.retcode in (ret_invalid_fill, getattr(mt5, "TRADE_RETCODE_INVALID", 10006)):
                alt = getattr(mt5, "ORDER_FILLING_FOK", 0) if (req["type_filling"] == getattr(mt5, "ORDER_FILLING_IOC", 1)) \
                      else getattr(mt5, "ORDER_FILLING_IOC", 1)
                r2 = mt5.order_send({**req, "type_filling": alt})
                if r2 and r2.retcode in (ret_ok, ret_placed, ret_partial):
                    try:
                        self.log.info(f"[MT5] order_send retry OK: retcode={r2.retcode} price={_sf(getattr(r2,'price',0.0)):.5f} ticket={getattr(r2,'order',getattr(r2,'deal',0))}")
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
                last_err = mt5.last_error() if hasattr(mt5, 'last_error') else (None, None)
                self.log.error(f"[MT5] order_send failed: retcode={getattr(r,'retcode','unknown')} last_error={last_err}")
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
    def _market_order_impl(self, instrument: str, side: int, lots: float) -> Dict[str, Any]:
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}
        return self._send_deal(instrument, side, lots)

    def _reduce_position_impl(self, instrument: str, lots: float, side: int) -> Dict[str, Any]:
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected"}
        # reduce == opposite side by same 'lots'
        return self._send_deal(instrument, side, lots)

    def _close_position_impl(self, instrument: str) -> Dict[str, Any]:
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
                side = -1 if getattr(p, "type", 1) == getattr(mt5, "POSITION_TYPE_BUY", 0) else 1
                lots = _sf(getattr(p, "volume", 0.0))
                if lots <= 0.0:
                    continue
                r = self._send_deal(instrument, side, lots)
                if not r.get("ok"):
                    return {"ok": False, "error": r.get("error", "close_failed")}
                last_px = _sf(r.get("price", last_px), last_px)

            return {"ok": True, "price": last_px}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def _sync_positions_impl(self) -> Dict[str, Dict[str, Any]]:
        """
        Net positions per symbol:
            side = sign(buy_lots - sell_lots)
            units = abs(net_lots) * contract_size
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

            # Default contract size from config
            default_cs = float(self.cfg.contract_size or 100_000.0)

            for sym, plist in by_sym.items():
                # Use symbol-specific contract size
                cs = self._get_contract_size(sym, default_cs)
                
                buy_lots = sum(_sf(getattr(p, "volume", 0.0)) for p in plist
                               if getattr(p, "type", 1) == getattr(mt5, "POSITION_TYPE_BUY", 0))
                sell_lots = sum(_sf(getattr(p, "volume", 0.0)) for p in plist
                                if getattr(p, "type", 1) == getattr(mt5, "POSITION_TYPE_SELL", 1))
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
                    return float(sum(v * px for v, px in zip(vols, prices)) / tot) if tot > 0 else 0.0

                entry_price = vwavg(plist, getattr(mt5, "POSITION_TYPE_BUY", 0)) if side > 0 \
                              else vwavg(plist, getattr(mt5, "POSITION_TYPE_SELL", 1))
                notional_eur = units * entry_price

                times: List[float] = []
                for q in plist:
                    t = getattr(q, "time", None)
                    if isinstance(t, (int, float)) and math.isfinite(t):
                        times.append(float(t))
                open_time = min(times) if times else 0.0

                out[sym] = {
                    "instrument": sym,
                    "side": side,
                    "units": float(units),
                    "entry_price": float(entry_price),
                    "notional_eur": float(notional_eur),
                    "open_time": open_time,
                }

            return out
        except Exception:
            return {}

    def fix_positions_without_sl_tp(self) -> Dict[str, Any]:
        """
        Check all open positions and add SL/TP if missing.
        This is critical for risk management if network disconnects.
        Returns dict with results for each position modified.
        """
        if not (_MT5 and self.connected):
            return {"ok": False, "error": "mt5_not_connected", "fixed": 0}
        
        try:
            # Check if fix_missing_sl_tp is enabled
            config = _load_sl_tp_config()
            if not config.get('fix_missing_sl_tp', True):
                return {"ok": True, "message": "fix_missing_sl_tp disabled", "fixed": 0}
            
            positions = list(mt5.positions_get() or [])
            if not positions:
                return {"ok": True, "message": "no_positions", "fixed": 0}
            
            fixed_count = 0
            results = []
            
            for pos in positions:
                ticket = getattr(pos, 'ticket', 0)
                symbol = getattr(pos, 'symbol', '')
                current_sl = _sf(getattr(pos, 'sl', 0.0))
                current_tp = _sf(getattr(pos, 'tp', 0.0))
                open_price = _sf(getattr(pos, 'price_open', 0.0))
                pos_type = getattr(pos, 'type', 0)
                
                # Check if SL or TP is missing
                needs_sl = current_sl <= 0 and config.get('auto_sl_enabled', True)
                needs_tp = current_tp <= 0 and config.get('auto_tp_enabled', True)
                
                if not needs_sl and not needs_tp:
                    continue
                
                # Get SL/TP pips for this symbol
                sl_pips, tp_pips = _get_sl_tp_pips(symbol)
                sl_distance = _pips_to_price(symbol, sl_pips)
                tp_distance = _pips_to_price(symbol, tp_pips)
                
                # Get symbol info for rounding
                symbol_info = mt5.symbol_info(symbol)
                digits = getattr(symbol_info, 'digits', 5) if symbol_info else 5
                
                # Calculate new SL/TP based on position type
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
                
                # Modify the position
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": symbol,
                    "position": ticket,
                    "sl": new_sl if new_sl > 0 else 0.0,
                    "tp": new_tp if new_tp > 0 else 0.0,
                }
                
                try:
                    self.log.info(f"[MT5] Fixing SL/TP for position {ticket}: {symbol} SL={new_sl:.5f} TP={new_tp:.5f}")
                except Exception:
                    pass
                
                result = mt5.order_send(request)
                
                if result and result.retcode in (getattr(mt5, "TRADE_RETCODE_DONE", 10009),):
                    fixed_count += 1
                    results.append({"ticket": ticket, "symbol": symbol, "sl": new_sl, "tp": new_tp, "ok": True})
                    try:
                        self.log.info(f"[MT5] Fixed position {ticket}: SL={new_sl:.5f} TP={new_tp:.5f}")
                    except Exception:
                        pass
                else:
                    error = getattr(result, 'retcode', 'unknown') if result else 'none'
                    results.append({"ticket": ticket, "symbol": symbol, "ok": False, "error": str(error)})
                    try:
                        self.log.warning(f"[MT5] Failed to fix position {ticket}: retcode={error}")
                    except Exception:
                        pass
            
            return {"ok": True, "fixed": fixed_count, "total": len(positions), "results": results}
        
        except Exception as e:
            try:
                self.log.error(f"[MT5] fix_positions_without_sl_tp exception: {e}")
            except Exception:
                pass
            return {"ok": False, "error": str(e), "fixed": 0}
