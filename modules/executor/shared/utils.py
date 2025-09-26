# modules/executor/shared/utils.py
from __future__ import annotations

import threading
from typing import Any, Dict, Optional, Iterable, Mapping, Tuple, TypeVar, Union
from decimal import Decimal, ROUND_HALF_UP, ROUND_FLOOR, ROUND_CEILING, InvalidOperation

T = TypeVar("T")

# ─────────────────────────────────────────────────────────
# Numeric utils
# ─────────────────────────────────────────────────────────
def _as_decimal(x: Union[int, float, str]) -> Decimal:
    try:
        return Decimal(str(x))
    except Exception:
        return Decimal(0)

def round_to_step(
    x: float,
    step: float,
    *,
    mode: str = "nearest",  # 'nearest' | 'down' | 'up'
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
) -> float:
    """
    Snap x to a multiple of `step` using Decimal (no float drift).

    mode:
      - 'nearest' : nearest, ties away from zero
      - 'down'    : toward zero
      - 'up'      : away from zero
    """
    try:
        if step <= 0:
            y = float(x)
        else:
            X = _as_decimal(x)
            S = _as_decimal(step)
            if S == 0:
                y = float(X)
            else:
                q = X / S
                if mode == "down":
                    q_rounded = q.to_integral_value(rounding=ROUND_FLOOR) if q >= 0 else q.to_integral_value(rounding=ROUND_CEILING)
                elif mode == "up":
                    q_rounded = q.to_integral_value(rounding=ROUND_CEILING) if q >= 0 else q.to_integral_value(rounding=ROUND_FLOOR)
                else:
                    # nearest, ties away from zero (works for +/-)
                    q_rounded = q.quantize(Decimal(1), rounding=ROUND_HALF_UP)
                y = float(q_rounded * S)

        if min_value is not None:
            y = max(y, float(min_value))
        if max_value is not None:
            y = min(y, float(max_value))
        return y
    except (InvalidOperation, Exception):
        y = round(x / step) * step if step > 0 else x
        if min_value is not None:
            y = max(y, float(min_value))
        if max_value is not None:
            y = min(y, float(max_value))
        return float(y)

# ─────────────────────────────────────────────────────────
# Instrument / symbol helpers
# ─────────────────────────────────────────────────────────
def canonical(inst: str) -> str:
    """Strict canonical form: strip non-alnum, uppercase."""
    if not inst:
        return ""
    out = "".join(ch for ch in inst if ch.isalnum())
    return out.upper()

def _fx_slash_form(inst: str) -> str:
    c = canonical(inst)
    return f"{c[:3]}/{c[3:]}" if len(c) == 6 else inst.upper().replace("_", "/")

def resolve_symbol(
    inst: str,
    overrides: Optional[Dict[str, str]] = None,
    *,
    broker: Optional[str] = None,
) -> str:
    """
    Resolve the runtime symbol (respect overrides, then broker defaults).
    mt5/metatrader -> 'EURUSD'
    oanda/fxcm     -> 'EUR_USD'
    else           -> canonical
    """
    if not inst:
        return ""
    if overrides:
        if inst in overrides:
            return overrides[inst]
        c = canonical(inst)
        if c in overrides:
            return overrides[c]

    b = (broker or "").lower()
    if b in {"mt5", "metatrader", "metatrader5"}:
        return canonical(inst)
    if b in {"oanda", "fxcm"}:
        return _fx_slash_form(inst)
    return canonical(inst)

# ─────────────────────────────────────────────────────────
# Safe bus adapter
# ─────────────────────────────────────────────────────────
class SafeBus:
    """
    Tiny adapter so we can call set/get safely from any thread,
    tolerant to different bus API signatures.
    """
    def __init__(self, bus: Any, default_module: str = "Executor"):
        self.bus = bus
        self._lock = threading.Lock()
        self._module = default_module

    def get(self, key: str, module: Optional[str] = None, default: Any = None) -> Any:
        if not self.bus:
            return default
        m = module or self._module
        try:
            return self.bus.get(key, m, default)  # (key, module, default)
        except TypeError:
            try:
                v = self.bus.get(key, m)         # (key, module)
                return v if v is not None else default
            except TypeError:
                try:
                    v = self.bus.get(key)        # (key)
                    return v if v is not None else default
                except Exception:
                    return default
        except Exception:
            return default

    def set(self, key: str, value: Any, thesis: str = "") -> None:
        if not self.bus:
            return
        try:
            with self._lock:
                try:
                    self.bus.set(key, value, module=self._module, thesis=thesis or key)  # (key, value, module=..., thesis=...)
                except TypeError:
                    self.bus.set(key, value)  # (key, value)
        except Exception:
            pass

    # list helpers
    def append(self, key: str, item: Any, thesis: str = "") -> None:
        with self._lock:
            cur = self.get(key, default=[])
            if not isinstance(cur, list):
                cur = []
            cur.append(item)
            self.set(key, cur, thesis or f"append:{key}")

    def extend(self, key: str, items: Iterable[Any], thesis: str = "") -> None:
        with self._lock:
            cur = self.get(key, default=[])
            if not isinstance(cur, list):
                cur = []
            cur.extend(list(items))
            self.set(key, cur, thesis or f"extend:{key}")

    # dict helper
    def merge(self, key: str, patch: Mapping[str, Any], thesis: str = "") -> None:
        with self._lock:
            cur = self.get(key, default={})
            if not isinstance(cur, dict):
                cur = {}
            merged = dict(cur)
            merged.update(dict(patch))
            self.set(key, merged, thesis or f"merge:{key}")

    def set_many(self, items: Iterable[Tuple[str, Any]], thesis_prefix: str = "") -> None:
        with self._lock:
            for k, v in items:
                self.set(k, v, f"{thesis_prefix}{k}" if thesis_prefix else k)
