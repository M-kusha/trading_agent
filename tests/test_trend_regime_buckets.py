"""Every trend regime bucket must be reachable.

_build_trades_with_regime banded the trend regime by |structure_trend|:

    > 0.5 -> strong_trend,  > 0.2 -> weak_trend,  else ranging

but structure_trend is a direction, not a magnitude. _compute_advanced_market_
structure emits exactly -1.0 (LL+LH), +1.0 (HH+HL) or 0.0, so nothing ever
landed in (0.2, 0.5] and weak_trend was mathematically unreachable. A live run
confirmed it: 1,104 strong_trend, 896 ranging, 0 weak_trend over 2,000 trades,
with the trend_following skill score computed across an always-empty bucket.

The continuous magnitude the band wanted is structure_strength, which is
produced alongside it and travels in the same entry context.
"""

from __future__ import annotations

import numpy as np
import pytest


def _regime(env, direction: float, strength: float) -> str:
    from envs.core.env_types import CloseReason, TradeResult

    result = TradeResult(
        net_pnl=10.0,
        initial_risk_eur=100.0,
        mae=-5.0,
        mfe=15.0,
        bars_held=4,
        close_reason=CloseReason.AGENT_CLOSE,
        entry_quality=0.5,
        direction="long",
        lot_size=0.1,
        entry_context={
            "structure_trend": direction,
            "structure_strength": strength,
            "volatility_regime": "medium",
            "risk_regime": "neutral",
        },
    )
    return env._build_trades_with_regime([result])[0]["trend_regime"]


def test_structure_trend_is_a_direction_not_a_magnitude():
    """Pin the producer's actual value domain - the band depended on it."""
    from envs.prop_firm.signals.market_structure import MarketStructureMixin

    import inspect

    src = inspect.getsource(MarketStructureMixin._compute_advanced_market_structure)
    assert "trend = 1.0" in src and "trend = -1.0" in src and "trend = 0.0" in src, (
        "structure_trend is no longer a three-valued sign; re-derive the regime "
        "bands against its new domain"
    )


def test_all_three_buckets_are_reachable(env):
    assert _regime(env, 0.0, 0.0) == "ranging"
    assert _regime(env, 1.0, 0.9) == "strong_trend"
    assert _regime(env, 1.0, 0.2) == "weak_trend"
    assert _regime(env, -1.0, 0.2) == "weak_trend"
    assert _regime(env, -1.0, 0.9) == "strong_trend"


def test_direction_zero_is_ranging_regardless_of_strength(env):
    """No directional structure means ranging even if the move was large."""
    assert _regime(env, 0.0, 1.0) == "ranging"


def test_weak_trend_is_not_empty_over_a_realistic_spread(env):
    """The regression was an empty bucket, so assert against a distribution."""
    rng = np.random.default_rng(0)
    counts = {"ranging": 0, "weak_trend": 0, "strong_trend": 0}
    for _ in range(300):
        direction = float(rng.choice([-1.0, 0.0, 1.0]))
        strength = float(rng.uniform(0.0, 1.0))
        counts[_regime(env, direction, strength)] += 1

    assert counts["weak_trend"] > 0, f"weak_trend still unreachable: {counts}"
    assert all(v > 0 for v in counts.values()), counts
