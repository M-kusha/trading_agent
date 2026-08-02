from __future__ import annotations

import numpy as np

KNOWN_DEAD_MAX = 12


def test_dead_dimension_count_within_budget(rollout):
    std = rollout.std(axis=0)
    dead = int((std < 1e-9).sum())
    assert dead <= KNOWN_DEAD_MAX, (
        f"{dead} constant dims exceeds the budget of {KNOWN_DEAD_MAX}. "
        "A previously-live observation dimension has gone dead."
    )


def test_market_and_account_blocks_are_fully_live(rollout):
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    for group in ("m15_price", "account"):
        start, end = FEATURE_GROUPS[group]
        std = rollout[:, start:end].std(axis=0)
        dead = int((std < 1e-9).sum())
        assert dead == 0, f"{dead} of {end - start} dims dead in '{group}'"


def test_volatility_proxy_spans_its_range(env):
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
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    fully_dead = []
    for group, (start, end) in FEATURE_GROUPS.items():
        std = rollout[:, start:end].std(axis=0)
        if int((std < 1e-9).sum()) == (end - start):
            fully_dead.append(group)
    assert not fully_dead, f"entire observation groups are constant: {fully_dead}"
