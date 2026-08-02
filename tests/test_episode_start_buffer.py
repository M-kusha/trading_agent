"""Episode starts must have enough history for every timeframe.

Regression for a bug that would have surfaced hours into a training run:
_episode_start_buffer counted only the M15 lookbacks and returned 230 bars,
ignoring the higher timeframes entirely. D1 is the binding constraint by a wide
margin - 40 D1 bars is 40 * (1440/15) = 3,840 M15 bars - so episodes sampled
near the start of the dataset raised

    ObservationContractError: D1.close must have >= 30 bars (strict). Got 24

intermittently, depending on which start the sampler happened to pick.
"""

from __future__ import annotations

import numpy as np

from envs.core.shared_utils import timeframe_to_minutes


def test_buffer_covers_the_deepest_timeframe(env):
    """The buffer must clear the D1 requirement, not just the M15 one."""
    buffer = env._episode_start_buffer()
    primary_minutes = max(1, env._tf_minutes())

    for tf, bars in env._HTF_LOOKBACK_BARS.items():
        ratio = max(1, timeframe_to_minutes(tf) // primary_minutes)
        required = bars * ratio
        assert buffer >= required, (
            f"buffer {buffer} bars is short of the {tf} requirement "
            f"({bars} bars x {ratio} = {required} primary bars)"
        )


def test_buffer_is_not_the_old_m15_only_value(env):
    """230 was the M15-only answer and is far too small."""
    assert env._episode_start_buffer() > 230


def test_resets_never_raise_across_many_seeds(env):
    """The failure was intermittent, so a single reset would not have caught it."""
    failures = []
    for seed in range(120):
        try:
            env.reset(seed=seed)
        except Exception as exc:  # noqa: BLE001 - the test is what it catches
            failures.append(f"seed {seed}: {type(exc).__name__}: {exc}")
    assert not failures, "resets failed:\n" + "\n".join(failures[:5])


def test_observation_is_valid_from_the_earliest_legal_start(env):
    """The worst case is the earliest start the sampler can choose."""
    env.reset(seed=0)
    env.current_step = env._episode_start_buffer()
    obs = env._get_observation()

    assert obs.shape == env.observation_space.shape
    assert not np.isnan(obs).any()
    assert float(np.abs(obs).sum()) > 0.0


def test_every_timeframe_supplies_its_contract_minimum(env):
    """Check the actual bar counts handed to the builder, not just the buffer."""
    from modules.meta.ppo_observation_builder import PPOObservationConfig

    cfg = PPOObservationConfig()
    env.reset(seed=0)
    env.current_step = env._episode_start_buffer()
    market_data = env._prepare_market_data(env._episode_instrument)

    assert len(market_data["M15"]["close"]) >= cfg.min_bars_m15
    for tf in ("H1", "H4", "D1"):
        supplied = len(market_data[tf]["close"])
        assert supplied >= cfg.min_bars_htf, (
            f"{tf} supplied {supplied} bars, contract requires {cfg.min_bars_htf}"
        )
