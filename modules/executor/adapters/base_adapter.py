
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LiveAdapterConfig:
    broker: str = "mt5"
    account_currency: str = "EUR"
    symbol_overrides: Optional[Dict[str, str]] = None


    lot_step: float = 0.01
    min_lot: float = 0.01
    contract_size: float = 100_000.0
    price_decimals: int = 5


    price_slippage: float = 0.0
    require_positive_lots: bool = True
    fallback_to_mid: bool = True


    max_retries: int = 2
    initial_backoff_ms: int = 150
    backoff_multiplier: float = 2.0
    rate_limit_per_sec: int = 15
    circuit_breaker_threshold: int = 6
    circuit_reset_sec: int = 30
    price_staleness_sec: int = 5


    telemetry_enabled: bool = True


def _now() -> float:
    return time.time()


def _sleep_ms(ms: int) -> None:
    time.sleep(max(0, ms) / 1000.0)


def _round_step(x: float, step: float) -> float:
    if step <= 0:
        return x
    return math.floor(x / step + 1e-12) * step


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
        return default
    except Exception:
        return default


def _open_time_to_ts(v: Any) -> float:
    if v is None:
        return 0.0


    if isinstance(v, (int, float)):
        ts = float(v)
    elif isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0

        try:
            ts = float(s)
        except Exception:

            try:
                from datetime import datetime

                dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
                ts = float(dt.timestamp())
            except Exception:
                return 0.0
    else:
        try:
            ts = float(v)
        except Exception:
            return 0.0

    if not math.isfinite(ts):
        return 0.0


    if ts > 1e12:
        ts /= 1000.0
    return ts


class BaseLiveAdapter:

    def __init__(self, cfg: LiveAdapterConfig):
        self.cfg = cfg
        self.connected: bool = False


        self._rl_last_sec: int = int(_now())
        self._rl_used: int = 0


        self._cb_failures: int = 0
        self._cb_opened_at: float = 0.0


        self._quote_ts: Dict[str, float] = {}


    def resolve_symbol(self, instrument: str) -> str:
        ov = self.cfg.symbol_overrides or {}
        return ov.get(instrument, instrument).replace("/", "").replace("_", "")

    def contract_size_for(self, instrument: str) -> float:
        sym = (instrument or "").upper().replace("_", "").replace("/", "")


        if "XAU" in sym or "GOLD" in sym:
            return 100.0
        if "XAG" in sym or "SILVER" in sym:
            return 5000.0


        if "BTC" in sym:
            return 1.0
        if "ETH" in sym:
            return 1.0


        cs = _safe_float(self.cfg.contract_size, 100_000.0)
        return cs if cs > 0 else 100_000.0

    def lots_to_units(self, lots: float, instrument: Optional[str] = None) -> float:
        cs = self.contract_size_for(instrument or "")
        return _safe_float(lots) * cs

    def units_to_lots(self, units: float, instrument: Optional[str] = None) -> float:
        cs = self.contract_size_for(instrument or "")
        if cs <= 0:
            return 0.0
        return _safe_float(units) / cs

    def round_lots(self, lots: float) -> float:
        lots = max(lots, 0.0)
        lots = _round_step(lots, max(self.cfg.lot_step, 1e-9))
        if lots > 0.0:
            lots = max(lots, self.cfg.min_lot)
        return lots

    def apply_fallback_slippage(self, side: int, price: float) -> float:
        slip = _safe_float(self.cfg.price_slippage, 0.0)
        if slip == 0.0:
            return price

        return price + (slip if side > 0 else -slip)


    def connect(self) -> bool:
        if self.connected:
            return True
        if self._circuit_open():
            return False

        ok = False
        delay = int(self.cfg.initial_backoff_ms)
        for attempt in range(self.cfg.max_retries + 1):
            ok = bool(self._connect_impl())
            if ok:
                self.connected = True
                self._cb_reset()
                break
            self._cb_note_failure()
            if attempt < self.cfg.max_retries:
                _sleep_ms(delay)
                delay = int(delay * self.cfg.backoff_multiplier)

        return ok

    def disconnect(self) -> None:
        try:
            self._disconnect_impl()
        finally:
            self.connected = False

    def is_connected(self) -> bool:
        return bool(self.connected)


    def get_account_info(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "balance": 0.0,
            "equity": 0.0,
            "margin": 0.0,
            "free_margin": 0.0,
            "margin_level": 0.0,
            "leverage": 0.0,
            "currency": self.cfg.account_currency,
        }
        if not self._ensure_ready():
            return out
        try:
            raw = self._get_account_info_impl() or {}

            for k in ("balance", "equity", "margin", "free_margin", "margin_level", "leverage"):
                out[k] = _safe_float(raw.get(k, out[k]), out[k])

            cur = raw.get("currency")
            if isinstance(cur, str) and cur:
                out["currency"] = cur
            return out
        except Exception:
            self._cb_note_failure()
            return out

    def get_prices(self, instrument: str) -> Dict[str, float]:
        if not self._ensure_ready():
            return {}
        sym = self.resolve_symbol(instrument)
        try:
            raw = self._get_prices_impl(sym) or {}
            bid = raw.get("bid")
            ask = raw.get("ask")
            mid = raw.get("mid")


            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (_safe_float(bid) + _safe_float(ask))
            if self.cfg.fallback_to_mid:
                if bid is None and mid is not None:
                    bid = float(mid)
                if ask is None and mid is not None:
                    ask = float(mid)
            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (float(bid) + float(ask))


            ts = raw.get("ts", _now())
            ts = _safe_float(ts, _now())
            self._quote_ts[sym] = ts


            if self.cfg.price_staleness_sec > 0 and (_now() - ts) > self.cfg.price_staleness_sec:
                return {}

            out: Dict[str, float] = {}
            if bid is not None:
                out["bid"] = _safe_float(bid)
            if ask is not None:
                out["ask"] = _safe_float(ask)
            if mid is not None:
                out["mid"] = _safe_float(mid)
            out["ts"] = ts
            return out
        except Exception:
            self._cb_note_failure()
            return {}


    def market_order(self, instrument: str, side: int, lots: float) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "ok": False,
            "instrument": instrument,
            "side": int(math.copysign(1, side)) if side != 0 else 0,
            "lots": 0.0,
            "price": 0.0,
        }
        if not self._ensure_ready():
            result["error"] = "not_connected_or_circuit_open"
            return result


        lots = self.round_lots(_safe_float(lots))
        if self.cfg.require_positive_lots and lots <= 0:
            result.update({"error": "non_positive_lots"})
            return result


        if not self._rate_ok():
            result.update({"error": "rate_limited"})
            return result

        sym = self.resolve_symbol(instrument)

        q = self.get_prices(instrument) or {}
        price_hint = q.get("mid") or q.get("bid") or q.get("ask") or 0.0

        delay = int(self.cfg.initial_backoff_ms)
        last_err: Optional[str] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._market_order_impl(
                    sym,
                    int(math.copysign(1, side)) if side != 0 else 0,
                    lots,
                ) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:
                    px = self.apply_fallback_slippage(side, price_hint)

                result.update(
                    {
                        "ok": ok,
                        "instrument": sym,
                        "side": int(math.copysign(1, side)) if side != 0 else 0,
                        "lots": lots,
                        "price": px,
                    }
                )
                if ok:
                    if "ticket" in raw:
                        result["ticket"] = raw["ticket"]
                    self._cb_reset()
                    return result
                last_err = raw.get("error", "unknown_error")
            except Exception as e:
                last_err = str(e)

            self._cb_note_failure()
            if attempt < self.cfg.max_retries:
                _sleep_ms(delay)
                delay = int(delay * self.cfg.backoff_multiplier)

        result["error"] = last_err or "execution_failed"
        return result

    def reduce_position(self, instrument: str, lots: float, side: int) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "ok": False,
            "instrument": instrument,
            "side": int(math.copysign(1, side)) if side != 0 else 0,
            "lots": 0.0,
            "price": 0.0,
        }
        if not self._ensure_ready():
            result["error"] = "not_connected_or_circuit_open"
            return result

        lots = self.round_lots(_safe_float(lots))
        if self.cfg.require_positive_lots and lots <= 0:
            result.update({"error": "non_positive_lots"})
            return result
        if not self._rate_ok():
            result.update({"error": "rate_limited"})
            return result

        sym = self.resolve_symbol(instrument)
        q = self.get_prices(instrument) or {}
        price_hint = q.get("mid") or q.get("bid") or q.get("ask") or 0.0

        delay = int(self.cfg.initial_backoff_ms)
        last_err: Optional[str] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._reduce_position_impl(
                    sym,
                    lots,
                    int(math.copysign(1, side)) if side != 0 else 0,
                ) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:
                    px = self.apply_fallback_slippage(side, price_hint)
                result.update(
                    {
                        "ok": ok,
                        "instrument": sym,
                        "side": int(math.copysign(1, side)) if side != 0 else 0,
                        "lots": lots,
                        "price": px,
                    }
                )
                if ok:
                    if "ticket" in raw:
                        result["ticket"] = raw["ticket"]
                    self._cb_reset()
                    return result
                last_err = raw.get("error", "unknown_error")
            except Exception as e:
                last_err = str(e)

            self._cb_note_failure()
            if attempt < self.cfg.max_retries:
                _sleep_ms(delay)
                delay = int(delay * self.cfg.backoff_multiplier)

        result["error"] = last_err or "execution_failed"
        return result

    def close_position(self, instrument: str) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": False, "instrument": instrument, "price": 0.0}
        if not self._ensure_ready():
            result["error"] = "not_connected_or_circuit_open"
            return result
        if not self._rate_ok():
            result["error"] = "rate_limited"
            return result

        sym = self.resolve_symbol(instrument)
        q = self.get_prices(instrument) or {}
        price_hint = q.get("mid") or q.get("bid") or q.get("ask") or 0.0

        delay = int(self.cfg.initial_backoff_ms)
        last_err: Optional[str] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._close_position_impl(sym) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:

                    px = self.apply_fallback_slippage(+1, price_hint)
                result.update({"ok": ok, "instrument": sym, "price": px})
                if ok:
                    if "ticket" in raw:
                        result["ticket"] = raw["ticket"]
                    self._cb_reset()
                    return result
                last_err = raw.get("error", "unknown_error")
            except Exception as e:
                last_err = str(e)

            self._cb_note_failure()
            if attempt < self.cfg.max_retries:
                _sleep_ms(delay)
                delay = int(delay * self.cfg.backoff_multiplier)

        result["error"] = last_err or "execution_failed"
        return result


    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"ok": False, "sl": 0.0, "tp": 0.0}
        if not self._ensure_ready():
            result["error"] = "not_connected_or_circuit_open"
            return result
        if not self._rate_ok():
            result["error"] = "rate_limited"
            return result

        delay = int(self.cfg.initial_backoff_ms)
        last_err: Optional[str] = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._modify_position_impl(ticket, sl, tp) or {}
                ok = bool(raw.get("ok", False))
                result.update({
                    "ok": ok,
                    "sl": _safe_float(raw.get("sl", sl or 0.0)),
                    "tp": _safe_float(raw.get("tp", tp or 0.0)),
                })
                if ok:
                    self._cb_reset()
                    return result
                last_err = raw.get("error", "unknown_error")
            except Exception as e:
                last_err = str(e)

            self._cb_note_failure()
            if attempt < self.cfg.max_retries:
                _sleep_ms(delay)
                delay = int(delay * self.cfg.backoff_multiplier)

        result["error"] = last_err or "modify_failed"
        return result


    def sync_positions(self) -> Dict[str, Dict[str, Any]]:
        if not self._ensure_ready():
            return {}
        try:
            raw = self._sync_positions_impl() or {}
            out: Dict[str, Dict[str, Any]] = {}
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        inst = str(k)
                        node = dict(v)
                        side_raw = _safe_float(node.get("side", 0))
                        side = int(math.copysign(1, side_raw)) if side_raw != 0 else 0
                        units = _safe_float(node.get("units", node.get("volume", 0.0)))
                        entry = _safe_float(node.get("entry_price", node.get("price", 0.0)))
                        notional = _safe_float(node.get("notional_eur", units * entry))
                        open_time_raw = node.get("open_time", node.get("time", None))
                        open_ts = _open_time_to_ts(open_time_raw)
                        age_hours = (
                            max(0.0, (_now() - open_ts) / 3600.0)
                            if open_ts > 0.0
                            else 0.0
                        )
                        out[inst] = {
                            "instrument": inst,
                            "side": side,
                            "units": units,
                            "entry_price": entry,
                            "notional_eur": notional,
                            "open_time": open_time_raw,
                            "age_hours": float(age_hours),

                            "unrealized_pnl": _safe_float(node.get("unrealized_pnl", node.get("profit", 0.0))),
                            "profit": _safe_float(node.get("profit", node.get("unrealized_pnl", 0.0))),
                            "current_price": _safe_float(node.get("current_price", node.get("price_current", 0.0))),
                            "price_current": _safe_float(node.get("price_current", node.get("current_price", 0.0))),
                            "ticket": int(node.get("ticket", 0) or 0),
                            "sl": _safe_float(node.get("sl", 0.0)),
                            "tp": _safe_float(node.get("tp", 0.0)),
                            "lots": _safe_float(node.get("lots", node.get("volume", 0.0))),
                        }
                    except Exception:
                        continue
            self._cb_reset()
            return out
        except Exception:
            self._cb_note_failure()
            return {}


    def _connect_impl(self) -> bool:
        raise NotImplementedError

    def _disconnect_impl(self) -> None:
        pass

    def _get_account_info_impl(self) -> Dict[str, float | str]:
        raise NotImplementedError

    def _get_prices_impl(self, instrument: str) -> Dict[str, float]:
        raise NotImplementedError

    def _market_order_impl(self, instrument: str, side: int, lots: float) -> Dict[str, Any]:
        raise NotImplementedError

    def _reduce_position_impl(self, instrument: str, lots: float, side: int) -> Dict[str, Any]:
        raise NotImplementedError

    def _close_position_impl(self, instrument: str) -> Dict[str, Any]:
        raise NotImplementedError

    def _sync_positions_impl(self) -> Dict[str, Dict[str, Any]]:
        raise NotImplementedError

    def _modify_position_impl(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


    def _rate_ok(self) -> bool:
        sec = int(_now())
        if sec != self._rl_last_sec:
            self._rl_last_sec = sec
            self._rl_used = 0
        if self._rl_used < max(1, int(self.cfg.rate_limit_per_sec)):
            self._rl_used += 1
            return True
        return False

    def _circuit_open(self) -> bool:
        if self._cb_opened_at <= 0.0:
            return False

        if (_now() - self._cb_opened_at) >= max(1, int(self.cfg.circuit_reset_sec)):

            return False
        return True

    def _cb_note_failure(self) -> None:
        self._cb_failures += 1
        if self._cb_failures >= max(1, int(self.cfg.circuit_breaker_threshold)):
            self._cb_opened_at = _now()

    def _cb_reset(self) -> None:
        self._cb_failures = 0
        self._cb_opened_at = 0.0

    def _ensure_ready(self) -> bool:
        if self.connected and not self._circuit_open():
            return True
        return self.connect()
