"""Train/live observation parity.

The original defect this guards against: training built its observation from
environment mixins while live built its own from bus keys. The two drifted, and
nothing detected it — a policy trained on one input would have met a different
one on its first live tick.

LiveStateHost removes the second code path by feeding live bars to the same
environment class. These tests assert that equality holds rather than assuming
it, which is the whole point of the design.
"""

from __future__ import annotations

import numpy as np
import pytest

from modules.meta.ppo_observation_builder import (
    PPO_OBS_SIZE,
    PPOObservationBuilder,
)
from trading.state import REQUIRED_BARS, LiveStateHost


@pytest.fixture(scope="module")
def frames(market_data):
    """A live-shaped rolling window taken from the tracked CSVs."""
    # Tail of each timeframe. Slicing every frame at one index would be wrong:
    # D1 holds 1,091 rows against M15's 99,908, so a shared cutoff empties it.
    # Taking the most recent bars also aligns the windows' end times naturally.
    bars = market_data["XAUUSD"]
    return {
        tf: bars[tf].tail(need * 3).reset_index(drop=True)
        for tf, need in REQUIRED_BARS.items()
    }


def test_host_produces_a_valid_observation(frames):
    host = LiveStateHost()
    host.update(frames)
    obs = PPOObservationBuilder().build(**host.observation_inputs())

    assert obs.shape == (PPO_OBS_SIZE,)
    assert not np.isnan(obs).any()
    assert float(np.abs(obs).sum()) > 0.0, "observation is all-zero"


def test_same_bars_give_the_same_observation(frames):
    """Deterministic replay: identical input, identical output, twice."""
    builder = PPOObservationBuilder()

    host_a = LiveStateHost()
    host_a.update(frames)
    obs_a = builder.build(**host_a.observation_inputs())

    host_b = LiveStateHost()
    host_b.update(frames)
    obs_b = builder.build(**host_b.observation_inputs())

    assert np.array_equal(obs_a, obs_b), "same bars produced different observations"


def test_live_host_matches_the_training_environment(frames, market_data):
    """THE parity assertion.

    A training env positioned on a given bar, and a live host fed the window
    ending at that same bar, must produce byte-identical observations. Any
    divergence here is the class of bug that made the original architecture
    unsafe to deploy.
    """
    from envs.core.env_types import PropFirmConfig
    from envs.prop_firm_env import PropFirmTradingEnv

    builder = PPOObservationBuilder()

    host = LiveStateHost()
    host.update(frames)
    live_obs = builder.build(**host.observation_inputs())

    # Training env over the identical frames, positioned on the identical bar.
    train_env = PropFirmTradingEnv({"XAUUSD": dict(frames)}, PropFirmConfig())
    train_env._episode_instrument = "XAUUSD"
    train_env.current_step = len(frames["M15"]) - 1
    train_obs = builder.build(
        market_data=train_env._prepare_market_data("XAUUSD"),
        expert_signals=train_env._prepare_expert_signals("XAUUSD"),
        risk_state=train_env._prepare_risk_state(),
        account_state=train_env._prepare_account_state("XAUUSD"),
        trading_mode_state=train_env._prepare_trading_mode_state("XAUUSD"),
        governor_state=train_env._get_governor_state(),
    )

    assert np.array_equal(live_obs, train_obs), (
        "live and training observations differ — the two paths have drifted"
    )


def test_short_history_is_rejected(frames):
    """A truncated window must raise, not silently produce a partial view."""
    truncated = dict(frames)
    truncated["M15"] = truncated["M15"].iloc[:20]

    host = LiveStateHost()
    with pytest.raises(ValueError, match="insufficient history"):
        host.update(truncated)


def test_missing_timeframe_is_rejected(frames):
    partial = {tf: df for tf, df in frames.items() if tf != "H4"}

    host = LiveStateHost()
    with pytest.raises(ValueError, match="missing timeframes"):
        host.update(partial)


def test_account_sync_reaches_the_observation(frames):
    """Broker balance must move the observation, or live trades on a fiction."""
    builder = PPOObservationBuilder()

    host = LiveStateHost()
    host.update(frames)
    baseline = builder.build(**host.observation_inputs())

    host.sync_account(balance=80_000.0, equity=80_000.0)
    drawn_down = builder.build(**host.observation_inputs())

    assert not np.array_equal(baseline, drawn_down), (
        "a 20% account drawdown did not change the observation"
    )
