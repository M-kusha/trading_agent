"""Data-integrity and look-ahead tests for the tracked feature CSVs.

The committed feature files were verified clean (every indicator aligns at
shift 0; max |corr| with next-bar return is 0.018). These tests pin that
property so a future change to the feature generator cannot silently introduce
look-ahead — the single defect most likely to produce a profitable-looking
backtest and a losing live account.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
PRIMARY = DATA_DIR / "XAUUSD_M15_features.csv"


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    if not PRIMARY.is_file():
        pytest.skip("XAUUSD_M15_features.csv not available")
    frame = pd.read_csv(PRIMARY, parse_dates=["time"])
    frame.columns = frame.columns.str.lower()
    return frame


def _best_shift(actual: pd.Series, recomputed: pd.Series) -> int:
    """Which time shift best explains the stored column?

    0 means causal. A negative shift means the column was computed from FUTURE
    bars, i.e. look-ahead.
    """
    scores = {}
    for shift in (-2, -1, 0, 1, 2):
        candidate = recomputed.shift(shift)
        valid = actual.notna() & candidate.notna()
        if not valid.any():
            scores[shift] = -1.0
            continue
        close = np.isclose(actual[valid], candidate[valid], rtol=1e-4, atol=1e-6)
        scores[shift] = float(close.mean())
    return max(scores, key=lambda k: scores[k])


@pytest.mark.parametrize(
    "column,builder",
    [
        ("sma_5", lambda d: d["close"].rolling(5).mean()),
        ("sma_20", lambda d: d["close"].rolling(20).mean()),
        ("sma_50", lambda d: d["close"].rolling(50).mean()),
        ("ema_12", lambda d: d["close"].ewm(span=12, adjust=False).mean()),
        ("ema_26", lambda d: d["close"].ewm(span=26, adjust=False).mean()),
        ("returns", lambda d: d["close"].pct_change()),
        ("momentum_5", lambda d: d["close"] / d["close"].shift(5) - 1.0),
        ("high_20", lambda d: d["high"].rolling(20).max()),
        ("low_20", lambda d: d["low"].rolling(20).min()),
        ("bb_middle", lambda d: d["close"].rolling(20).mean()),
    ],
)
def test_indicator_is_causal(df, column, builder):
    """Each indicator must be best explained by shift 0, never a future shift."""
    if column not in df.columns:
        pytest.skip(f"{column} not in feature set")
    shift = _best_shift(df[column], builder(df))
    assert shift >= 0, f"{column} best matches shift {shift} -> LOOK-AHEAD"
    assert shift == 0, f"{column} best matches shift {shift}, expected 0"


def test_no_feature_strongly_predicts_next_bar_return(df):
    """A leaked column shows an implausibly high correlation with the future."""
    forward = df["close"].shift(-1) / df["close"] - 1.0
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corrwith(forward).abs().dropna()
    worst = corr.sort_values(ascending=False).head(1)
    name, value = worst.index[0], float(worst.iloc[0])
    assert value < 0.10, f"'{name}' correlates {value:.3f} with next-bar return - probable leak"


def test_ohlc_relationships_hold(df):
    high, low = df["high"].to_numpy(), df["low"].to_numpy()
    open_, close = df["open"].to_numpy(), df["close"].to_numpy()
    assert not (high < low).any(), "high < low"
    assert not ((open_ > high) | (open_ < low)).any(), "open outside [low, high]"
    assert not ((close > high) | (close < low)).any(), "close outside [low, high]"


def test_timestamps_are_unique_and_monotonic(df):
    time = df["time"]
    assert not time.duplicated().any(), "duplicate timestamps"
    deltas = time.diff().dt.total_seconds().dropna()
    assert (deltas > 0).all(), "timestamps are not strictly increasing"


def test_no_nan_or_non_positive_prices(df):
    assert df[["open", "high", "low", "close"]].notna().all().all(), "NaN in OHLC"
    assert (df["close"] > 0).all(), "non-positive close price"


def test_no_weekend_bars(df):
    """Saturday bars, or Sunday before the 22:00 open, indicate a bad merge."""
    time = df["time"]
    weekend = ((time.dt.weekday == 5) | ((time.dt.weekday == 6) & (time.dt.hour < 22))).sum()
    assert int(weekend) == 0, f"{weekend} weekend bars present"


def test_higher_timeframe_agrees_with_aggregated_m15(df):
    """H1 must equal the aggregate of its four M15 bars, or the timeframes were
    built from different sources."""
    h1_path = DATA_DIR / "XAUUSD_H1_features.csv"
    if not h1_path.is_file():
        pytest.skip("H1 file not available")
    h1 = pd.read_csv(h1_path, parse_dates=["time"])
    h1.columns = h1.columns.str.lower()

    m15 = df.copy()
    m15["hour_key"] = m15["time"].dt.floor("h")
    agg = m15.groupby("hour_key").agg(
        m_open=("open", "first"), m_high=("high", "max"),
        m_low=("low", "min"), m_close=("close", "last"), n=("close", "size"),
    )
    joined = h1.set_index("time").join(agg, how="inner")
    complete = joined[joined["n"] == 4]
    assert len(complete) > 1000, "too few complete hours to validate"

    for col, ref in (("open", "m_open"), ("high", "m_high"),
                     ("low", "m_low"), ("close", "m_close")):
        diff = (complete[col] - complete[ref]).abs().max()
        assert diff < 1e-4, f"H1 {col} disagrees with aggregated M15 by {diff}"
