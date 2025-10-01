from __future__ import annotations

import datetime
from typing import Optional, Iterable, Tuple


def normalize_session_name(name: Optional[str]) -> str:
    """Normalize session aliases to canonical: asian/european/american/rollover/unknown."""
    try:
        if not name:
            return 'unknown'
        s = str(name).strip().lower()
        mapping = {
            # American
            'us': 'american', 'americas': 'american', 'ny': 'american', 'new_york': 'american', 'new-york': 'american',
            'american': 'american', 'usa': 'american', 'na': 'american', 'america': 'american',
            # European
            'eu': 'european', 'europe': 'european', 'london': 'european', 'uk': 'european', 'gb': 'european',
            'european': 'european',
            # Asian
            'asia': 'asian', 'apac': 'asian', 'tokyo': 'asian', 'jp': 'asian', 'japan': 'asian', 'sydney': 'asian',
            'asian': 'asian',
            # Closed windows (rollover/maintenance/overnight)
            'roll': 'closed', 'rollover': 'closed', 'overnight': 'closed', 'closed': 'closed',
            # Unknown/other
            'unknown': 'unknown', 'weekend': 'weekend', 'holiday': 'holiday',
        }
        return mapping.get(s, s)
    except Exception:
        return 'unknown'


def _in_window(hour: int, start: int, end: int) -> bool:
    return (start <= hour < end) if start < end else (hour >= start or hour < end)


def classify_session(
    *,
    hour: int,
    weekend: bool,
    closed_windows: Iterable[Tuple[int, int]] = ((21, 23), (11, 12)),
) -> str:
    """Classify canonical session by UTC hour and weekend flag.

    - Returns one of: 'asian', 'european', 'american', 'closed'.
    - closed_windows: iterable of (start_hour, end_hour) to treat as closed (rollover/maintenance).
    - Weekend always maps to 'closed'.
    """
    try:
        h = int(hour)
        if weekend:
            return 'closed'
        for start, end in closed_windows:
            if _in_window(h, int(start), int(end)):
                return 'closed'
        if 0 <= h < 7:
            return 'asian'
        if 7 <= h < 12:
            return 'european'
        if 12 <= h < 20:
            return 'american'
        return 'closed'
    except Exception:
        return 'unknown'


def infer_market_session(dt: datetime.datetime | None = None) -> str:
    """Heuristic UTC-based session inference → canonical, using default closed windows."""
    try:
        now = dt or datetime.datetime.utcnow()
        wk = now.weekday() in (5, 6)
        return classify_session(hour=now.hour, weekend=wk)
    except Exception:
        return 'unknown'
