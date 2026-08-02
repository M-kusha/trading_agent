from __future__ import annotations

from typing import Any, Dict

import numpy as np


def sanitize_value(v: Any) -> Any:
    try:
        if isinstance(v, (int, float, np.generic)):
            f = float(v)
            return None if not np.isfinite(f) else f
        return v
    except Exception:
        return None


def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (metrics or {}).items():
        out[k] = sanitize_value(v)
    return out
