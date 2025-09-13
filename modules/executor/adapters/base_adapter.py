from __future__ import annotations

import time
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple, Callable


# ─────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────
@dataclass
class LiveAdapterConfig:
    broker: str = "mt5"
    account_currency: str = "EUR"
    symbol_overrides: Optional[Dict[str, str]] = None

    # contract sizing
    lot_step: float = 0.01
    min_lot: float = 0.01
    contract_size: float = 100_000.0  # units per 1.0 lot (FX)
    price_decimals: int = 5

    # execution policy
    price_slippage: float = 0.0       # fallback slippage in price units if broker returns no price
    require_positive_lots: bool = True
    fallback_to_mid: bool = True      # if bid/ask missing, derive mid

    # reliability
    max_retries: int = 2
    initial_backoff_ms: int = 150
    backoff_multiplier: float = 2.0
    rate_limit_per_sec: int = 15      # coarse token bucket
    circuit_breaker_threshold: int = 6
    circuit_reset_sec: int = 30
    price_staleness_sec: int = 5      # reject stale quotes if older than this

    # telemetry (best-effort, adapter-agnostic)
    telemetry_enabled: bool = True


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────
# Base Interface + Robust Wrappers
# ─────────────────────────────────────────────────────────
class BaseLiveAdapter:
    """
    Adapter contract for live execution providers (e.g., MT5).

    Implement the *_impl methods in concrete adapters:
      - _connect_impl() -> bool
      - _disconnect_impl() -> None
      - _get_account_info_impl() -> Dict[str, float]
      - _get_prices_impl(instrument) -> Dict[str, float]  # {'bid','ask','mid','ts'}
      - _market_order_impl(instrument, side, lots) -> Dict[str, Any]
      - _reduce_position_impl(instrument, lots, side) -> Dict[str, Any]
      - _close_position_impl(instrument) -> Dict[str, Any]
      - _sync_positions_impl() -> Dict[str, Dict[str, Any]]

    Public methods (used by Executor) run through robust wrappers providing:
      - rate limiting, retries + backoff, circuit breaking
      - lot rounding/clamping
      - quote freshness checks & price normalization
      - fallback slippage if broker omits execution price
      - standardized return shapes
    """

    def __init__(self, cfg: LiveAdapterConfig):
        self.cfg = cfg
        self.connected: bool = False

        # rate limiter state
        self._rl_last_sec: int = int(_now())
        self._rl_used: int = 0

        # circuit breaker state
        self._cb_failures: int = 0
        self._cb_opened_at: float = 0.0  # 0 => closed

        # last quotes timestamps by symbol for freshness checks
        self._quote_ts: Dict[str, float] = {}

    # ─────────────────────────────────────────────────────
    # Symbol / sizing utilities
    # ─────────────────────────────────────────────────────
    def resolve_symbol(self, instrument: str) -> str:
        """Apply overrides like {'XAU/USD': 'XAUUSD'} before hitting the broker."""
        ov = self.cfg.symbol_overrides or {}
        return ov.get(instrument, instrument).replace("/", "").replace("_", "")

    def lots_to_units(self, lots: float) -> float:
        return _safe_float(lots) * _safe_float(self.cfg.contract_size, 100_000.0)

    def units_to_lots(self, units: float) -> float:
        cs = _safe_float(self.cfg.contract_size, 100_000.0)
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
        """If broker didn't return an execution price, bias by slippage."""
        slip = _safe_float(self.cfg.price_slippage, 0.0)
        if slip == 0.0:
            return price
        # buy => pay slightly more; sell => receive slightly less
        return price + (slip if side > 0 else -slip)

    # ─────────────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────────────
    def connect(self) -> bool:
        """Robust connect with retries + circuit breaker semantics."""
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

    # ─────────────────────────────────────────────────────
    # Account & market data
    # ─────────────────────────────────────────────────────
    def get_account_info(self) -> Dict[str, float]:
        """Return {'balance','equity','margin','free_margin','margin_level'} — missing keys default to 0.0."""
        out = {"balance": 0.0, "equity": 0.0, "margin": 0.0, "free_margin": 0.0, "margin_level": 0.0}
        if not self._ensure_ready():
            return out
        try:
            raw = self._get_account_info_impl() or {}
            for k in out.keys():
                out[k] = _safe_float(raw.get(k, out[k]), out[k])
            return out
        except Exception:
            self._cb_note_failure()
            return out

    def get_prices(self, instrument: str) -> Dict[str, float]:
        """
        Return {'bid','ask','mid','ts'}.
        - Fills missing bid/ask from mid when allowed (fallback_to_mid).
        - Enforces staleness: if ts older than price_staleness_sec, returns {}.
        """
        if not self._ensure_ready():
            return {}
        sym = self.resolve_symbol(instrument)
        try:
            raw = self._get_prices_impl(sym) or {}
            bid = raw.get("bid")
            ask = raw.get("ask")
            mid = raw.get("mid")

            # derive mid/bid/ask if needed
            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (_safe_float(bid) + _safe_float(ask))
            if self.cfg.fallback_to_mid:
                if bid is None and mid is not None:
                    bid = float(mid)
                if ask is None and mid is not None:
                    ask = float(mid)
            if mid is None and bid is not None and ask is not None:
                mid = 0.5 * (float(bid) + float(ask))

            # timestamp
            ts = raw.get("ts", _now())
            ts = _safe_float(ts, _now())
            self._quote_ts[sym] = ts

            # staleness
            if self.cfg.price_staleness_sec > 0 and (_now() - ts) > self.cfg.price_staleness_sec:
                return {}

            out = {}
            if bid is not None: out["bid"] = _safe_float(bid)
            if ask is not None: out["ask"] = _safe_float(ask)
            if mid is not None: out["mid"] = _safe_float(mid)
            out["ts"] = ts
            return out
        except Exception:
            self._cb_note_failure()
            return {}

    # ─────────────────────────────────────────────────────
    # Execution (robust wrappers)
    # ─────────────────────────────────────────────────────
    def market_order(self, instrument: str, side: int, lots: float) -> Dict[str, Any]:
        """
        Place a market order. Return:
          {'ok': bool, 'instrument': str, 'side': int, 'lots': float, 'price': float, 'ticket': Any? , 'error': str?}
        """
        result = {"ok": False, "instrument": instrument, "side": int(math.copysign(1, side)) if side != 0 else 0, "lots": 0.0, "price": 0.0}
        if not self._ensure_ready():
            result["error"] = "not_connected_or_circuit_open"
            return result

        # lot policy
        lots = self.round_lots(_safe_float(lots))
        if self.cfg.require_positive_lots and lots <= 0:
            result.update({"error": "non_positive_lots"})
            return result

        # rate limit
        if not self._rate_ok():
            result.update({"error": "rate_limited"})
            return result

        sym = self.resolve_symbol(instrument)
        # try to fetch a price for fallback slippage if broker omits executed price
        q = self.get_prices(instrument) or {}
        price_hint = q.get("mid") or q.get("bid") or q.get("ask") or 0.0

        delay = int(self.cfg.initial_backoff_ms)
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._market_order_impl(sym, int(math.copysign(1, side)) if side != 0 else 0, lots) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:
                    px = self.apply_fallback_slippage(side, price_hint)

                result.update({
                    "ok": ok,
                    "instrument": sym,
                    "side": int(math.copysign(1, side)) if side != 0 else 0,
                    "lots": lots,
                    "price": px,
                })
                if ok:
                    if "ticket" in raw: result["ticket"] = raw["ticket"]
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
        """
        Reduce an existing position (close part of it).
        Return shape mirrors market_order().
        """
        result = {"ok": False, "instrument": instrument, "side": int(math.copysign(1, side)) if side != 0 else 0, "lots": 0.0, "price": 0.0}
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
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._reduce_position_impl(sym, lots, int(math.copysign(1, side)) if side != 0 else 0) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:
                    px = self.apply_fallback_slippage(side, price_hint)
                result.update({"ok": ok, "instrument": sym, "side": int(math.copysign(1, side)) if side != 0 else 0, "lots": lots, "price": px})
                if ok:
                    if "ticket" in raw: result["ticket"] = raw["ticket"]
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
        """
        Close an entire position in 'instrument'.
        Return {'ok', 'instrument', 'price', 'ticket'?, 'error'?}
        """
        result = {"ok": False, "instrument": instrument, "price": 0.0}
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
        last_err = None
        for attempt in range(self.cfg.max_retries + 1):
            try:
                raw = self._close_position_impl(sym) or {}
                ok = bool(raw.get("ok", False))
                px = _safe_float(raw.get("price", 0.0))
                if px <= 0.0 and price_hint > 0.0:
                    px = self.apply_fallback_slippage(+1, price_hint)  # side doesn’t matter much for full close
                result.update({"ok": ok, "instrument": sym, "price": px})
                if ok:
                    if "ticket" in raw: result["ticket"] = raw["ticket"]
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

    # ─────────────────────────────────────────────────────
    # Positions snapshot
    # ─────────────────────────────────────────────────────
    def sync_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Map: {
          'EURUSD': {
             'instrument': 'EURUSD',
             'side': +1|-1,
             'units': float,
             'entry_price': float,
             'notional_eur': float,
             'open_time': iso|unix (optional)
          }, ...
        }
        """
        if not self._ensure_ready():
            return {}
        try:
            raw = self._sync_positions_impl() or {}
            out: Dict[str, Dict[str, Any]] = {}
            for k, v in (raw.items() if isinstance(raw, dict) else []):
                try:
                    inst = str(k)
                    node = dict(v)
                    side = int(math.copysign(1, _safe_float(node.get("side", 0)))) if _safe_float(node.get("side", 0)) != 0 else 0
                    units = _safe_float(node.get("units", node.get("volume", 0.0)))
                    entry = _safe_float(node.get("entry_price", node.get("price", 0.0)))
                    notional = _safe_float(node.get("notional_eur", units * entry))
                    out[inst] = {
                        "instrument": inst,
                        "side": side,
                        "units": units,
                        "entry_price": entry,
                        "notional_eur": notional,
                        "open_time": node.get("open_time", node.get("time", None)),
                    }
                except Exception:
                    continue
            self._cb_reset()
            return out
        except Exception:
            self._cb_note_failure()
            return {}

    # ─────────────────────────────────────────────────────
    # Abstract impls (to be provided by concrete adapter)
    # ─────────────────────────────────────────────────────
    def _connect_impl(self) -> bool:
        raise NotImplementedError

    def _disconnect_impl(self) -> None:
        pass

    def _get_account_info_impl(self) -> Dict[str, float]:
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

    # ─────────────────────────────────────────────────────
    # Internals: rate limit & circuit breaker
    # ─────────────────────────────────────────────────────
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
        # half-open after reset window
        if (_now() - self._cb_opened_at) >= max(1, int(self.cfg.circuit_reset_sec)):
            # half-open: allow a try, but keep state until success
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
