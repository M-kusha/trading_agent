"""Behavioral coverage for chronology, sampling, and raw-evaluation boundaries.

These tests intentionally use a timestamp column with a RangeIndex.  That is
the shape produced by the repository's CSV loaders and is the boundary where
timezone/session regressions have previously hidden.
"""

from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pandas as pd
import pytest

from envs.core.env_types import PropFirmConfig
from envs.curriculum import CurriculumManager, CurriculumStage, DataDifficulty
from envs.prop_firm_env import PropFirmTradingEnv
from train.train_prop_firm import (
    _primary_frame,
    build_walk_forward_folds,
    create_curriculum_env,
    split_data_by_time,
    train_curriculum_agent,
)


CUT = pd.Timestamp("2026-06-15T00:00:00Z")


def _ohlcv_frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    x = np.arange(len(times), dtype=np.float64)
    close = 2_000.0 + 0.012 * x + 0.35 * np.sin(x / 37.0)
    open_ = close + 0.04 * np.sin(x / 11.0)
    high = np.maximum(open_, close) + 0.20
    low = np.minimum(open_, close) - 0.20
    return pd.DataFrame(
        {
            "time": times,
            "open": open_.astype(np.float32),
            "high": high.astype(np.float32),
            "low": low.astype(np.float32),
            "close": close.astype(np.float32),
            "volume": (100.0 + x % 23).astype(np.float32),
            "spread": (30.0 + times.hour.to_numpy() % 7).astype(np.float32),
        }
    ).reset_index(drop=True)


@pytest.fixture(scope="module")
def boundary_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    m15_times = pd.date_range(
        "2026-04-15T00:00:00Z", periods=8_000, freq="15min"
    )
    start, end = m15_times[0], m15_times[-1]
    frames = {
        "M15": _ohlcv_frame(m15_times),
        "H1": _ohlcv_frame(pd.date_range(start, end, freq="1h")),
        "H4": _ohlcv_frame(pd.date_range(start, end, freq="4h")),
        "D1": _ohlcv_frame(pd.date_range(start.floor("D"), end.ceil("D"), freq="1D")),
    }
    return {"XAUUSD": frames}


def _plain_env(
    data: Dict[str, Dict[str, pd.DataFrame]],
    *,
    max_steps: int = 128,
    start_min: str | None = None,
) -> PropFirmTradingEnv:
    config = PropFirmConfig(
        max_steps_per_episode=max_steps,
        episode_start_min_time=start_min,
        mirror_augmentation_prob=0.0,
        high_vol_oversample_prob=0.0,
        domain_randomization_enabled=False,
    )
    config.execution.latency_bars = 0
    return PropFirmTradingEnv(data, config)


def _manager() -> CurriculumManager:
    return CurriculumManager(
        initial_stage=CurriculumStage.EXPLORER,
        auto_promote=False,
        auto_demote=False,
        verbose=False,
        rng_seed=7,
    )


def test_explicit_june_split_keeps_context_but_never_labels_before_cut(
    boundary_data, monkeypatch
):
    train, holdout, split_ts = split_data_by_time(
        boundary_data,
        ratio=0.15,
        split_at=CUT.isoformat(),
        include_holdout_context=True,
    )

    assert split_ts is not None
    assert pd.Timestamp(split_ts) == CUT
    for timeframe in ("M15", "H1", "H4", "D1"):
        train_times = pd.to_datetime(train["XAUUSD"][timeframe]["time"], utc=True)
        holdout_frame = holdout["XAUUSD"][timeframe]
        holdout_times = pd.to_datetime(holdout_frame["time"], utc=True)
        assert train_times.max() < CUT
        assert holdout_times.min() < CUT <= holdout_times.max()
        assert pd.Timestamp(holdout_frame.attrs["episode_start_min_time"]) == CUT

    env = _plain_env(holdout, start_min=holdout["XAUUSD"]["M15"].attrs["episode_start_min_time"])
    monkeypatch.setattr(env, "_sample_episode_start", lambda low, _high: low)
    obs, _ = env.reset(seed=11)
    start = pd.Timestamp(holdout["XAUUSD"]["M15"].iloc[env.current_step]["time"])

    assert start >= CUT, f"earliest eligible episode started before the holdout cut: {start}"
    assert obs.shape == env.observation_space.shape
    assert np.isfinite(obs).all()


def test_requested_but_too_short_holdout_fails_instead_of_aliasing(boundary_data):
    with pytest.raises(ValueError, match="requested holdout is not executable"):
        train_curriculum_agent(
            data=boundary_data,
            total_timesteps=1,
            n_envs=1,
            learning_rate=3e-4,
            batch_size=32,
            n_steps=64,
            n_epochs=1,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=0.03,
            policy_hidden=32,
            value_hidden=32,
            checkpoint_freq=1_000,
            holdout_ratio=0.15,
            holdout_split_at="2026-07-01T00:00:00Z",
        )


def test_latest_episode_start_is_reachable_without_tail_history_reserve(
    boundary_data, monkeypatch
):
    env = _plain_env(boundary_data)
    sampled: dict[str, int] = {}

    def choose_latest(low: int, high: int) -> int:
        sampled.update(low=low, high=high)
        return high - 1

    monkeypatch.setattr(env, "_sample_episode_start", choose_latest)
    env.reset(seed=13)

    assert sampled["low"] == env._episode_start_buffer()
    assert sampled["high"] == len(boundary_data["XAUUSD"]["M15"]) - env.config.max_steps_per_episode
    assert env.current_step == sampled["high"] - 1


def test_last_walk_forward_fold_reaches_end_and_starts_labels_at_boundary(
    boundary_data, monkeypatch
):
    folds = build_walk_forward_folds(
        boundary_data, n_folds=3, val_ratio=0.15, min_train_ratio=0.55
    )
    assert len(folds) == 3

    train, validation = folds[-1]
    train_primary = _primary_frame(train["XAUUSD"])
    validation_primary = _primary_frame(validation["XAUUSD"])
    assert train_primary is not None and validation_primary is not None

    source_primary = boundary_data["XAUUSD"]["M15"]
    boundary = pd.Timestamp(validation_primary.attrs["episode_start_min_time"])
    train_times = pd.to_datetime(train_primary["time"], utc=True)
    validation_times = pd.to_datetime(validation_primary["time"], utc=True)
    source_times = pd.to_datetime(source_primary["time"], utc=True)
    eligible_times = validation_times[validation_times >= boundary]

    assert validation_times.iloc[-1] == source_times.iloc[-1]
    assert eligible_times.iloc[0] == boundary
    assert train_times.max() < boundary
    assert set(train_times).isdisjoint(set(eligible_times))

    env = _plain_env(validation, start_min=validation_primary.attrs["episode_start_min_time"])
    monkeypatch.setattr(env, "_sample_episode_start", lambda low, _high: low)
    env.reset(seed=17)
    sampled_start = pd.Timestamp(validation_primary.iloc[env.current_step]["time"])
    assert sampled_start >= boundary


def test_curriculum_installs_data_difficulty(boundary_data):
    manager = _manager()
    env = create_curriculum_env(
        boundary_data,
        manager,
        seed=19,
        use_action_masking=False,
    )

    assert env._data_difficulty is not None
    assert env._data_difficulty is not manager.stage_config.data_difficulty
    assert env._data_difficulty == manager.stage_config.data_difficulty
    assert env._valid_start_indices is not None
    assert len(env._valid_start_indices) > 0


def test_rangeindex_time_column_drives_session_filter_and_empty_policy_fails_closed(
    boundary_data,
):
    env = _plain_env(boundary_data)
    london_only = DataDifficulty(
        include_asian_session=False,
        include_london_session=True,
        include_ny_session=False,
        include_overlap_sessions=False,
    )
    env.set_data_difficulty(london_only)

    primary = boundary_data["XAUUSD"]["M15"]
    selected_hours = pd.to_datetime(
        primary.iloc[env._valid_start_indices]["time"], utc=True
    ).dt.hour
    assert selected_hours.between(8, 15).all()

    no_sessions = DataDifficulty(
        include_asian_session=False,
        include_london_session=False,
        include_ny_session=False,
        include_overlap_sessions=False,
    )
    with pytest.raises(ValueError, match="0 valid indices"):
        env.set_data_difficulty(no_sessions)

    with pytest.raises(ValueError, match="unsupported DataDifficulty regimes"):
        env.set_data_difficulty(DataDifficulty(allowed_regimes=["made_up_regime"]))


def test_temporal_regime_sampling_assigns_the_declared_probability_mass(boundary_data):
    env = _plain_env(boundary_data)
    env.config.recent_regime_start_time = CUT.isoformat()
    env.config.recent_regime_target_share = 0.40
    env.set_data_difficulty(DataDifficulty())
    env.np_random = np.random.default_rng(2026)
    low, high = env._episode_sampling_bounds(env._episode_start_buffer(), 0)

    starts = np.asarray(
        [env._sample_episode_start_with_difficulty(low, high) for _ in range(5_000)],
        dtype=np.int64,
    )
    times = pd.to_datetime(boundary_data["XAUUSD"]["M15"].iloc[starts]["time"], utc=True)
    observed = float((times >= CUT).mean())
    assert observed == pytest.approx(0.40, abs=0.025)


def test_temporal_regime_sampling_rejects_misaligned_timestamp_contract(
    boundary_data, monkeypatch
):
    env = _plain_env(boundary_data)
    env.config.recent_regime_start_time = CUT.isoformat()
    env.config.recent_regime_target_share = 0.40
    env.set_data_difficulty(DataDifficulty())
    low, high = env._episode_sampling_bounds(env._episode_start_buffer(), 0)
    monkeypatch.setattr(env, "_df_time_ns", lambda _frame: np.asarray(42, dtype=np.int64))

    with pytest.raises(ValueError, match="one-dimensional timestamp array"):
        env._sample_episode_start_with_difficulty(low, high)


def test_raw_evaluation_disables_all_stochastic_augmentation(boundary_data):
    env = create_curriculum_env(
        boundary_data,
        _manager(),
        seed=23,
        use_action_masking=False,
        raw_evaluation_mode=True,
    )

    assert env.config.raw_evaluation_mode is True
    assert env.config.domain_randomization_enabled is False
    assert env.config.mirror_augmentation_prob == 0.0
    assert env.config.high_vol_oversample_prob == 0.0
    assert env.config.execution.deterministic_costs is True
    assert env.config.execution.rejection_enabled is False
    assert env.config.execution.spread_shock_enabled is False
    assert env._data_difficulty is None
    assert env._mirror_active is False
    assert env._episode_spread_mult == pytest.approx(1.0)
    assert env._episode_slip_mult == pytest.approx(1.0)


def test_named_validation_filter_survives_raw_reset_and_then_clears(boundary_data):
    env = create_curriculum_env(
        boundary_data,
        _manager(),
        seed=29,
        use_action_masking=False,
        raw_evaluation_mode=True,
    )
    high_volatility = DataDifficulty(volatility_percentile_range=(0.70, 1.0))

    env.set_scenario_data_difficulty(high_volatility)
    env.reset(seed=29)
    assert env._data_difficulty == high_volatility
    assert env._data_difficulty is not high_volatility
    assert env._valid_start_indices is not None
    assert len(env._valid_start_indices) > 0
    assert env._volatility_percentiles is not None
    selected_percentiles = env._volatility_percentiles[env._valid_start_indices]
    assert np.all(selected_percentiles >= 0.70)
    assert np.all(selected_percentiles <= 1.0)

    env.set_scenario_data_difficulty(None)
    env.reset(seed=30)
    assert env._data_difficulty is None
    assert env._valid_start_indices is None


def test_validation_start_schedule_is_disjoint_and_exhaustion_fails_loudly(
    boundary_data,
):
    env = create_curriculum_env(
        boundary_data,
        _manager(),
        seed=31,
        use_action_masking=False,
        raw_evaluation_mode=True,
    )

    starts = env.configure_validation_episode_starts(
        min_episodes=10,
        max_episodes=12,
    )

    assert len(starts) == 12
    assert starts == sorted(starts)
    assert all(b - a > 168 for a, b in zip(starts, starts[1:]))

    # One internal sentinel handles DummyVecEnv's automatic post-terminal reset.
    for expected in [*starts, starts[0]]:
        env.reset()
        assert env.current_step == expected
    with pytest.raises(RuntimeError, match="schedule exhausted"):
        env.reset()

    env.clear_validation_episode_starts()
    assert env._validation_start_schedule is None


def test_episode_counters_and_identity_are_episode_scoped(boundary_data):
    env = _plain_env(boundary_data)
    env.reset(seed=31)
    env._mask_decision_steps = 17
    env._mask_collapse_steps = 6
    env._stop_mode_steps = 4

    env.reset(seed=32)
    stats = env.get_episode_stats()
    expected_start = pd.Timestamp(
        boundary_data["XAUUSD"]["M15"].iloc[env.current_step]["time"]
    )

    # reset exposes one initial decision state to the policy.  The important
    # invariant is that the prior episode's 17/6/4 counts were discarded.
    assert env._mask_decision_steps == 1
    assert env._mask_collapse_steps <= 1
    assert env._stop_mode_steps <= 1
    assert stats["episode_start_index"] == env.current_step
    assert pd.Timestamp(stats["episode_start_time"]) == expected_start
    assert stats["episode_length"] == 0
    assert stats["mask_decision_steps"] == 1


def test_reset_fails_loudly_when_primary_timestamp_is_missing(boundary_data):
    malformed = copy.deepcopy(boundary_data)
    malformed["XAUUSD"]["M15"] = malformed["XAUUSD"]["M15"].drop(columns=["time"])
    env = _plain_env(malformed)

    with pytest.raises(RuntimeError, match="no valid market-data timestamp"):
        env.reset(seed=33)


def test_effective_rehearsal_stage_drives_runtime_config_and_metadata(
    boundary_data, monkeypatch
):
    manager = CurriculumManager(
        initial_stage=CurriculumStage.TREND_STUDENT,
        auto_promote=False,
        auto_demote=False,
        verbose=False,
        rng_seed=37,
    )
    monkeypatch.setattr(
        manager, "get_effective_stage", lambda: CurriculumStage.EXPLORER
    )
    env = create_curriculum_env(
        boundary_data,
        manager,
        seed=37,
        use_action_masking=False,
    )

    env.reset(seed=37)
    stats = env.get_episode_stats()

    assert manager.current_stage == CurriculumStage.TREND_STUDENT
    assert env._curriculum_stage_idx == CurriculumStage.EXPLORER.value
    assert stats["effective_curriculum_stage"] == "EXPLORER"
    assert env._data_difficulty == manager.get_effective_stage_config().data_difficulty


def test_invalid_curriculum_override_fails_before_episode(boundary_data, monkeypatch):
    manager = _manager()
    env = create_curriculum_env(
        boundary_data,
        manager,
        seed=41,
        use_action_masking=False,
    )
    invalid = copy.deepcopy(manager.stage_config)
    invalid.reward_overrides = dict(invalid.reward_overrides or {})
    invalid.reward_overrides["field_that_does_not_exist"] = 1.0
    monkeypatch.setattr(manager, "get_effective_stage_config", lambda: invalid)

    with pytest.raises(RuntimeError, match="partially applied stage"):
        env.reset(seed=41)
