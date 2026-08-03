"""An M15 decision may not see a higher-timeframe bar that has not closed.

This was the most serious defect in the project and it survived an explicit
look-ahead audit.

Higher-timeframe rows are stamped with the bar's OPEN. Selecting the latest row
whose timestamp is <= the current M15 time therefore returned the bar that is
still forming: at M15 17:00 the agent read an H1 close and high determined by
17:45 data. On D1 the leak reaches 23h45m. Every training run, walk-forward
fold, holdout and backtest produced before the fix consumed it.

Note what did NOT catch it. train/live parity passed throughout, because both
paths leaked identically - parity proves two implementations agree, not that
either is causal. Observation-health passed too, because leaked values have a
perfectly ordinary distribution. Only asking what a timestamp means finds this.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

TIMEFRAME_MINUTES = {"H1": 60, "H4": 240, "D1": 1440}


def test_higher_timeframe_rows_are_stamped_with_the_bar_open(market_data):
    """Pin the fact the whole bug rests on.

    If a data source ever switches to close-stamping, the causality arithmetic
    must be revisited rather than silently becoming conservative.
    """
    m15 = market_data["XAUUSD"]["M15"].copy()
    h1 = market_data["XAUUSD"]["H1"].copy()
    m15["time"] = pd.to_datetime(m15["time"])
    h1["time"] = pd.to_datetime(h1["time"])

    row = h1.iloc[5000]
    start = row["time"]
    forward = m15[(m15["time"] >= start) & (m15["time"] < start + pd.Timedelta(hours=1))]

    assert len(forward) > 0
    assert float(row["close"]) == pytest.approx(float(forward["close"].iloc[-1])), (
        "H1 rows are no longer open-stamped; re-derive the causality bound"
    )
    assert float(row["high"]) == pytest.approx(float(forward["high"].max()))


@pytest.mark.parametrize("timeframe", ["H1", "H4", "D1"])
def test_the_agent_never_sees_an_unfinished_higher_timeframe_bar(env, market_data, timeframe):
    """The load-bearing assertion, checked at real decision points."""
    m15 = market_data["XAUUSD"]["M15"].copy()
    m15["time"] = pd.to_datetime(m15["time"])
    htf = market_data["XAUUSD"][timeframe].copy()
    htf["time"] = pd.to_datetime(htf["time"])
    span = pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])

    env.reset(seed=0)
    env._episode_instrument = "XAUUSD"

    checked = 0
    for step in range(20_000, 20_400, 7):
        env.current_step = step
        decision_open = m15["time"].iloc[step]
        # The decision is taken at the close of the current M15 bar.
        decision_time = decision_open + pd.Timedelta(minutes=15)

        seen = env._get_ohlcv("XAUUSD", lookback=3, timeframe=timeframe)
        if not seen or len(seen.get("close", [])) == 0:
            continue

        # Derive the newest bar that HAS closed by the decision time, rather than
        # searching for the observed close in the frame: gold revisits price
        # levels, so matching by value picks up an unrelated bar months away.
        closed = htf[htf["time"] + span <= decision_time]
        if closed.empty:
            continue
        expected_open = closed["time"].iloc[-1]
        expected_close = float(closed["close"].iloc[-1])

        assert float(seen["close"][-1]) == pytest.approx(expected_close), (
            f"{timeframe}: at decision time {decision_time} the agent should see the "
            f"bar opening {expected_open} (closing {expected_open + span}), but the "
            f"observed close does not match it"
        )
        checked += 1

    assert checked > 0, "no decision points were actually verified"


def test_the_boundary_case_is_the_one_that_leaked(env, market_data):
    """At an exact H1 boundary the forming bar must be excluded, not included."""
    m15 = market_data["XAUUSD"]["M15"].copy()
    m15["time"] = pd.to_datetime(m15["time"])
    h1 = market_data["XAUUSD"]["H1"].copy()
    h1["time"] = pd.to_datetime(h1["time"])

    env.reset(seed=0)
    env._episode_instrument = "XAUUSD"

    step = next(
        i for i in range(20_000, 20_200) if m15["time"].iloc[i].minute == 0
    )
    env.current_step = step
    boundary = m15["time"].iloc[step]

    seen = env._get_ohlcv("XAUUSD", lookback=3, timeframe="H1")
    forming = h1[h1["time"] == boundary]
    assert not forming.empty, "fixture does not contain the forming bar"

    assert float(seen["close"][-1]) != pytest.approx(float(forming["close"].iloc[0])), (
        "the agent sees the H1 bar stamped at the decision time - that bar closes "
        "45 minutes in the future"
    )


def test_observations_are_finite_after_the_causal_shift(env):
    """Stepping back a whole bar must not starve the observation contract."""
    env.reset(seed=0)
    for _ in range(50):
        obs = env._get_observation()
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        _o, _r, term, trunc, _i = env.step(int(valid[0]) if valid.size else 0)
        if term or trunc:
            break
