"""Observation-health budget.

test_obs_contract.py proves the observation is alive. This file pins *how*
alive it is, so silent decay is a test failure rather than a slow leak.

Current state (measured 2026-08-02, 700-step rollout): 10 of 90 dims are
constant, down from 25 of 106 before the v6.0 schema change. Each remaining one
is listed below with its cause. The budget exists so a change that kills further
dimensions has to say so.
"""
from __future__ import annotations

import numpy as np

# Dims known to be constant, with the reason. Anything outside this set that
# goes dead is a regression.
#
# The v5.8 expert_raw block (16 dims, 15 of them constant) was removed entirely
#   in v6.0 - it read the modules/voting/experts proposal schema while training
#   feeds envs/prop_firm/signals, and the signals behind it showed no measured
#   skill against an always-long control.
# seasonality dims: the training seasonality expert is an explicit stub.
# structure_bias / is_trained / trading_mode: heuristics that rarely or never
#   change state over a single rollout.
KNOWN_DEAD_MAX = 12


def test_dead_dimension_count_within_budget(rollout):
    std = rollout.std(axis=0)
    dead = int((std < 1e-9).sum())
    assert dead <= KNOWN_DEAD_MAX, (
        f"{dead} constant dims exceeds the budget of {KNOWN_DEAD_MAX}. "
        "A previously-live observation dimension has gone dead."
    )


def test_market_and_account_blocks_are_fully_live(rollout):
    """These blocks are derived directly from price and account state; a
    constant dim here always indicates a bug, never a design choice."""
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    for group in ("m15_price", "account"):
        start, end = FEATURE_GROUPS[group]
        std = rollout[:, start:end].std(axis=0)
        dead = int((std < 1e-9).sum())
        assert dead == 0, f"{dead} of {end - start} dims dead in '{group}'"


def test_volatility_proxy_spans_its_range(env):
    """Regression cover for the mis-scaled volatility proxy.

    `clip(std(returns) * 100, 0, 1)` is calibrated for ~1% per-bar volatility;
    XAUUSD M15 runs ~0.07%, so the proxy sat below 0.3 on ~99% of bars and the
    'high' state (> 0.7) was mathematically unreachable. Both `vol_state` and
    `zone_type` were therefore constant.
    """
    env._episode_instrument = "XAUUSD"
    values = []
    for step in range(1000, 60000, 997):
        env.current_step = step
        values.append(env._atr_vol_proxy("XAUUSD"))
    arr = np.asarray(values, dtype=np.float64)

    assert arr.std() > 0.05, "volatility proxy is effectively constant"
    low = float((arr < 0.3).mean())
    high = float((arr > 0.7).mean())
    assert low < 0.90, f"'low' volatility state covers {low:.1%} of bars"
    assert high > 0.005, f"'high' volatility state is unreachable ({high:.3%} of bars)"


def test_no_dimension_is_constant_zero_across_a_whole_group(rollout):
    """A fully dead group means its producer is disconnected, not merely quiet."""
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    fully_dead = []
    for group, (start, end) in FEATURE_GROUPS.items():
        std = rollout[:, start:end].std(axis=0)
        if int((std < 1e-9).sum()) == (end - start):
            fully_dead.append(group)
    assert not fully_dead, f"entire observation groups are constant: {fully_dead}"
