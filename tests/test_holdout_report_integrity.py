from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from envs.core.env_types import PropFirmConfig
from envs.prop_firm_env import PropFirmTradingEnv
from evaluate.holdout_report import (
    AlwaysFlat,
    Result,
    _json_safe,
    build_non_overlapping_episode_starts,
    load_bound_model_provenance,
    run_policy,
)
from modules.meta.ppo_observation_builder import PPO_OBS_VERSION


def _frame(times: pd.DatetimeIndex) -> pd.DataFrame:
    x = np.arange(len(times), dtype=np.float64)
    close = 2_000.0 + np.sin(x / 31.0)
    return pd.DataFrame(
        {
            "time": times,
            "open": close,
            "high": close + 0.25,
            "low": close - 0.25,
            "close": close,
            "volume": 100.0 + x % 17,
            "spread": 30.0 + x % 3,
        }
    )


@pytest.fixture(scope="module")
def evaluation_data():
    m15 = pd.date_range("2026-01-01T00:00:00Z", periods=5_000, freq="15min")
    start, end = m15[0], m15[-1]
    return {
        "XAUUSD": {
            "M15": _frame(m15),
            "H1": _frame(pd.date_range(start, end, freq="1h")),
            "H4": _frame(pd.date_range(start, end, freq="4h")),
            "D1": _frame(pd.date_range(start.floor("D"), end.ceil("D"), freq="1D")),
        }
    }


def test_holdout_schedule_is_paired_and_non_overlapping(evaluation_data):
    cfg = PropFirmConfig(max_steps_per_episode=64)
    cfg.mirror_augmentation_prob = 0.0
    cfg.high_vol_oversample_prob = 0.0
    cfg.domain_randomization_enabled = False
    env = PropFirmTradingEnv(evaluation_data, cfg)
    starts = build_non_overlapping_episode_starts(
        env,
        requested_episodes=8,
        max_steps=64,
    )
    env.close()

    assert len(starts) == 8
    assert starts == sorted(starts)
    assert all(b - a > 64 for a, b in zip(starts, starts[1:]))

    result = run_policy(
        AlwaysFlat(),
        evaluation_data,
        cfg,
        episodes=8,
        seed=99,
        max_steps=64,
        episode_starts=starts,
    )
    assert result.episodes == 8
    assert result.independent_windows == 8
    assert result.episode_start_indices == starts
    assert [row["start_time"] for row in result.episode_records] == sorted(
        row["start_time"] for row in result.episode_records
    )


def test_unbound_or_hash_mismatched_model_fails_closed(tmp_path):
    model = tmp_path / "candidate.zip"
    model.write_bytes(b"candidate-model")

    with pytest.raises(FileNotFoundError, match="provenance is required"):
        load_bound_model_provenance(model, allow_unbound=False)

    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    provenance = {
        "model": {"sha256": digest},
        "observation_schema": {"version": PPO_OBS_VERSION},
    }
    model.with_suffix(".provenance.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )
    assert load_bound_model_provenance(model, allow_unbound=False) == provenance

    model.write_bytes(b"changed-after-provenance")
    with pytest.raises(ValueError, match="hash does not match"):
        load_bound_model_provenance(model, allow_unbound=False)


def test_report_json_replaces_non_finite_diagnostics_with_null():
    assert _json_safe({"pf": float("inf"), "ci": float("-inf")}) == {
        "pf": None,
        "ci": None,
    }


def test_development_report_never_labels_thresholds_as_a_pass():
    result = Result(
        name="synthetic",
        return_pct=12.0,
        max_drawdown_pct=2.0,
        worst_daily_dd_pct=1.0,
    )

    assert result.challenge_thresholds_met is True
    assert result.challenge_threshold_diagnostic == "THRESHOLDS_MET_DIAGNOSTIC_ONLY"
    assert "PASS" not in result.challenge_threshold_diagnostic
