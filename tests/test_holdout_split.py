"""The holdout must actually exist, and must cut every timeframe together.

The split was index-based off _min_len_across(data), which is the D1 row count
(1,091) rather than M15's 99,908, and slice_data_by_index applied that same
index range to every timeframe - so a split at index 927 would have cut M15
down to 927 of its 99,908 bars. A length guard caught that the numbers made no
sense and disabled the holdout entirely, so every run logged

    Holdout disabled: insufficient bars for requested holdout (n=1091, ...)

and trained and evaluated on identical data. There was no out-of-sample check
anywhere in the pipeline.

Splitting on a timestamp keeps the timeframes aligned: each is cut wherever
that instant falls in its own index.
"""

from __future__ import annotations

import pytest

from train.train_prop_firm import _primary_frame, split_data_by_time

RATIO = 0.15


def test_the_primary_frame_is_the_finest_timeframe(market_data):
    primary = _primary_frame(market_data["XAUUSD"])
    assert primary is not None
    assert len(primary) == len(market_data["XAUUSD"]["M15"]), (
        "primary must be M15, not the smallest frame - sizing off D1 is what "
        "disabled the holdout"
    )


def test_the_split_produces_a_real_holdout(market_data):
    train, holdout, split_ts = split_data_by_time(market_data, RATIO)

    assert split_ts is not None
    tr = _primary_frame(train["XAUUSD"])
    ho = _primary_frame(holdout["XAUUSD"])
    assert tr is not None and ho is not None
    assert len(ho) > 1000, f"holdout is only {len(ho)} bars"
    assert len(tr) > len(ho), "train must be larger than holdout"

    total = len(market_data["XAUUSD"]["M15"])
    assert len(ho) / total == pytest.approx(RATIO, abs=0.02)


def test_every_timeframe_is_cut_at_the_same_instant(market_data):
    train, holdout, split_ts = split_data_by_time(market_data, RATIO)

    for tf in ("M15", "H1", "H4", "D1"):
        tr = train["XAUUSD"][tf]
        ho = holdout["XAUUSD"][tf]
        assert len(tr) > 0, f"{tf} train side is empty"
        assert len(ho) > 0, f"{tf} holdout side is empty"
        assert tr["time"].iloc[-1] < split_ts <= ho["time"].iloc[0], (
            f"{tf} is not cut at the split timestamp"
        )


def test_no_timeframe_is_truncated_to_the_smallest_frames_length(market_data):
    """The specific corruption index-based slicing would have caused."""
    train, _holdout, _ts = split_data_by_time(market_data, RATIO)

    d1_len = len(market_data["XAUUSD"]["D1"])
    m15_train = len(train["XAUUSD"]["M15"])
    assert m15_train > d1_len * 10, (
        f"M15 train is {m15_train} bars, near the D1 length of {d1_len} - the "
        f"split is being sized off the wrong timeframe"
    )


def test_train_and_holdout_do_not_overlap(market_data):
    train, holdout, _ts = split_data_by_time(market_data, RATIO)

    for tf in ("M15", "H1", "H4", "D1"):
        tr_times = set(train["XAUUSD"][tf]["time"])
        ho_times = set(holdout["XAUUSD"][tf]["time"])
        assert not (tr_times & ho_times), f"{tf} train and holdout share bars"


def test_the_holdout_is_chronologically_after_training(market_data):
    """A shuffled split would leak the future into training."""
    train, holdout, _ts = split_data_by_time(market_data, RATIO)

    for tf in ("M15", "H1", "H4", "D1"):
        assert train["XAUUSD"][tf]["time"].iloc[-1] < holdout["XAUUSD"][tf]["time"].iloc[0], (
            f"{tf} holdout is not strictly after train"
        )
