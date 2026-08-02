"""Direction must be symmetric, or the agent learns "gold goes up".

XAUUSD rose 149.8% across this dataset (1,735 -> 4,332). A random long held 96
bars earns +241 points and a random short loses exactly that, so a live run
showed long P&L of +26,553 against short -17,591 with no policy skill involved
at all. The holdout is also a bull market (+32.8%), and envs/curriculum
contains no holdout references, so nothing anywhere in the pipeline punishes a
permanently-long policy: a leveraged buy-and-hold passes every gate.

Mirroring half the episodes (p' = anchor - p, high and low swapped) makes an
uptrend an identical downtrend, so edge has to come from structure.

The reflection must be linear rather than log-space: linear negates every bar
move exactly and preserves absolute point moves, which is what P&L depends on.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_mirroring_negates_every_bar_move(env):
    frames = env._base_data["XAUUSD"]
    mirrored = env._build_mirrored_data()["XAUUSD"]

    for tf in ("M15", "H1", "H4", "D1"):
        orig_close = frames[tf]["close"].to_numpy(dtype=float)
        mirr_close = mirrored[tf]["close"].to_numpy(dtype=float)

        d_orig = np.diff(orig_close)
        d_mirr = np.diff(mirr_close)
        assert np.allclose(d_mirr, -d_orig, atol=1e-6), f"{tf} moves not exactly negated"


def test_an_uptrend_becomes_a_downtrend(env):
    close = env._base_data["XAUUSD"]["M15"]["close"].to_numpy(dtype=float)
    mirr = env._build_mirrored_data()["XAUUSD"]["M15"]["close"].to_numpy(dtype=float)

    assert close[-1] > close[0], "fixture data is expected to trend up"
    assert mirr[-1] < mirr[0], "mirrored data must trend down"
    assert (close[-1] - close[0]) == pytest.approx(-(mirr[-1] - mirr[0]), rel=1e-9)


def test_high_and_low_are_swapped_not_just_shifted(env):
    frames = env._base_data["XAUUSD"]["M15"]
    mirrored = env._build_mirrored_data()["XAUUSD"]["M15"]

    high = frames["high"].to_numpy(dtype=float)
    low = frames["low"].to_numpy(dtype=float)
    m_high = mirrored["high"].to_numpy(dtype=float)
    m_low = mirrored["low"].to_numpy(dtype=float)

    anchor = env._mirror_anchors["XAUUSD"]
    assert np.allclose(m_high, anchor - low, atol=1e-6)
    assert np.allclose(m_low, anchor - high, atol=1e-6)
    # The invariant that would break bar logic if the swap were omitted.
    assert np.all(m_high >= m_low - 1e-9), "mirrored high must stay above low"


def test_mirrored_prices_stay_positive(env):
    mirrored = env._build_mirrored_data()["XAUUSD"]
    for tf, df in mirrored.items():
        for col in ("open", "high", "low", "close"):
            v = df[col].to_numpy(dtype=float)
            assert np.all(v > 0), f"{tf}.{col} went non-positive under mirroring"


def test_volatility_is_preserved(env):
    """Mirroring must change direction only, not the character of the market."""
    close = env._base_data["XAUUSD"]["M15"]["close"].to_numpy(dtype=float)
    mirr = env._build_mirrored_data()["XAUUSD"]["M15"]["close"].to_numpy(dtype=float)

    assert np.std(np.diff(close)) == pytest.approx(np.std(np.diff(mirr)), rel=1e-9)


def test_execution_and_observation_use_the_same_prices(env):
    """_get_price_mid reads the frame directly, _get_ohlcv slices it.

    If mirroring reached one path and not the other, fills would disagree with
    what the agent saw. Swapping self.data wholesale is what prevents that, so
    assert the two paths agree in both modes.
    """
    for mirror_on in (False, True):
        env.reset(seed=3)
        env._mirror_active = not mirror_on
        env.config.mirror_augmentation_prob = 1.0 if mirror_on else 0.0
        env._select_episode_data()

        mid = env._get_price_mid("XAUUSD")
        ohlcv = env._get_ohlcv("XAUUSD", lookback=2)
        assert ohlcv, "no ohlcv returned"
        assert mid == pytest.approx(float(ohlcv["close"][-1]), rel=1e-9), (
            f"execution price and observation price disagree (mirror={mirror_on})"
        )


def test_both_modes_are_reachable_and_episodes_still_run(env):
    env.config.mirror_augmentation_prob = 0.5
    seen = set()
    for seed in range(40):
        env.reset(seed=seed)
        seen.add(env._mirror_active)
        obs = env._get_observation()
        assert not np.isnan(obs).any(), f"NaN observation with mirror={env._mirror_active}"
    assert seen == {True, False}, f"only reached {seen} in 40 resets"


def test_disabling_the_probability_keeps_the_original_data(env):
    env.config.mirror_augmentation_prob = 0.0
    env._mirror_active = False
    for seed in range(10):
        env.reset(seed=seed)
        assert env._mirror_active is False
        assert env.data is env._base_data
