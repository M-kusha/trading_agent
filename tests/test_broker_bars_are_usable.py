"""Broker bars carry only OHLCV, and that has to be enough.

data/processed CSVs have 58 columns - returns, rsi, atr, sma_*, macd and so on.
Bars pulled from MT5 have open/high/low/close/volume/spread and nothing else, so
after appending them 51 of the 58 columns are entirely NaN on the newer rows.

That is only safe because the env reads none of those columns: _get_ohlcv
resolves open/high/low/close plus volume and spread, _prepare_market_data hands
the observation builder OHLCV alone, and the indicators the signals use (adx,
rsi, macd) are computed in code from that OHLCV rather than read from the file.

If anything ever starts reading a precomputed column, training silently begins
consuming NaN on the most recent - and most relevant - bars. These tests fail
loudly instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from envs.core.env_types import PropFirmConfig
from envs.prop_firm_env import PropFirmTradingEnv

BROKER_COLUMNS = {"time", "open", "high", "low", "close", "volume", "spread"}


@pytest.fixture(scope="module")
def merged_env():
    from train.train_prop_firm import load_market_data

    data = load_market_data(instruments=["XAUUSD"], min_bars=5000)
    return PropFirmTradingEnv(data, PropFirmConfig()), data


def test_the_merge_actually_happened(merged_env):
    _env, data = merged_env
    m15 = data["XAUUSD"]["M15"]
    assert len(m15) > 100_000, f"only {len(m15)} bars - broker bars were not appended"
    assert m15["time"].max() > pd.Timestamp("2026-01-01")


def test_appended_rows_have_complete_ohlcv(merged_env):
    _env, data = merged_env
    m15 = data["XAUUSD"]["M15"]
    recent = m15.loc[m15["time"] > pd.Timestamp("2026-01-01")]
    assert len(recent) > 1000

    for col in ("open", "high", "low", "close", "volume"):
        assert not recent[col].isna().any(), f"{col} has NaN in the appended rows"
        assert (recent[col] >= 0).all(), f"{col} has negative values"

    assert (recent["high"] >= recent["low"]).all()


def test_observations_stay_finite_inside_the_appended_region(merged_env):
    """The decisive check: step the env where the engineered columns are NaN."""
    env, data = merged_env
    m15 = data["XAUUSD"]["M15"]
    first_new = int((m15["time"] <= pd.Timestamp("2025-12-18 23:45")).sum())

    env.reset(seed=0)
    env.current_step = first_new + 2000

    rng = np.random.default_rng(0)
    for _ in range(200):
        obs = env._get_observation()
        assert not np.isnan(obs).any(), "NaN reached the observation on appended bars"
        assert not np.isinf(obs).any(), "Inf reached the observation on appended bars"

        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        _obs, _r, term, trunc, _info = env.step(int(rng.choice(valid)) if valid.size else 0)
        if term or trunc:
            break


def test_trades_execute_on_appended_bars(merged_env):
    """Finite observations are not enough - the execution path must work too."""
    env, data = merged_env
    m15 = data["XAUUSD"]["M15"]
    first_new = int((m15["time"] <= pd.Timestamp("2025-12-18 23:45")).sum())

    env.reset(seed=1)
    env.current_step = first_new + 2000

    rng = np.random.default_rng(1)
    for _ in range(600):
        mask = env.action_masks()
        valid = np.flatnonzero(mask)
        _obs, _r, term, trunc, _info = env.step(int(rng.choice(valid)) if valid.size else 0)
        if term or trunc:
            break

    stats = env.get_episode_stats()
    assert int(stats.get("trade_count", 0)) > 0, "no trade executed on appended bars"


def test_the_env_reads_only_broker_available_columns():
    """Pin the contract that makes the merge safe."""
    import inspect

    src = inspect.getsource(PropFirmTradingEnv._get_ohlcv)
    for col in ("open", "high", "low", "close"):
        assert f'"{col}"' in src

    forbidden = ("rsi", "macd", "sma_", "atr_percent", "log_returns", "volatility_10")
    for name in forbidden:
        assert f'_resolve_column(df, "{name}")' not in src, (
            f"_get_ohlcv now reads the precomputed column {name!r}, which is NaN on "
            f"every broker-appended bar"
        )
