"""The observation-health panel must detect a blind agent.

During the outage where every observation was np.zeros(90), the dashboard
rendered normal reward and win-rate curves for the entire run. Nothing on
screen distinguished a blind agent from a learning one, and the failure was
found by reading source, not by monitoring.

These tests assert the detection works, using injected faults rather than
waiting for a real one.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from modules.meta.ppo_observation_builder import PPO_OBS_SIZE, PPO_OBS_VERSION


def _callback_with(samples):
    """A VecEpisodeTradingCallback with a pre-loaded observation sample buffer."""
    from train.callbacks.episode_callback import VecEpisodeTradingCallback

    cb = VecEpisodeTradingCallback.__new__(VecEpisodeTradingCallback)
    cb._obs_samples = deque(samples, maxlen=512)
    cb._obs_health = {}
    return cb


def test_detects_a_constant_observation():
    """THE canary. zeros(N) every step is exactly what happened before."""
    cb = _callback_with([np.zeros(PPO_OBS_SIZE) for _ in range(64)])
    health = cb._compute_obs_health()

    assert health["status"] == "blind"
    assert "cannot see" in health["alert"]
    assert health["dead_dims"] == PPO_OBS_SIZE


def test_detects_nan_contamination():
    rng = np.random.default_rng(0)
    samples = [rng.normal(size=PPO_OBS_SIZE) for _ in range(64)]
    samples[10][3] = np.nan
    health = _callback_with(samples)._compute_obs_health()

    assert health["status"] == "bad"
    assert health["nan_count"] >= 1


def test_detects_mostly_dead_observation():
    """More than half the dimensions constant is a broken producer."""
    rng = np.random.default_rng(0)
    samples = []
    for _ in range(64):
        row = np.zeros(PPO_OBS_SIZE)
        row[: PPO_OBS_SIZE // 4] = rng.normal(size=PPO_OBS_SIZE // 4)
        samples.append(row)
    health = _callback_with(samples)._compute_obs_health()

    assert health["status"] == "bad"
    assert health["dead_dims"] > PPO_OBS_SIZE // 2


def test_healthy_observation_passes():
    rng = np.random.default_rng(0)
    health = _callback_with([rng.normal(size=PPO_OBS_SIZE) for _ in range(64)])._compute_obs_health()

    assert health["status"] == "good"
    assert health["dead_dims"] == 0
    assert health["nan_count"] == 0


def test_reports_schema_identity():
    """A checkpoint/builder mismatch must be visible, not inferred."""
    rng = np.random.default_rng(0)
    health = _callback_with([rng.normal(size=PPO_OBS_SIZE) for _ in range(64)])._compute_obs_health()

    assert health["schema_version"] == PPO_OBS_VERSION
    assert health["schema_size"] == PPO_OBS_SIZE
    assert len(health["schema_hash"]) == 12


def test_per_block_breakdown_is_reported():
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    rng = np.random.default_rng(0)
    health = _callback_with([rng.normal(size=PPO_OBS_SIZE) for _ in range(64)])._compute_obs_health()

    assert set(health["blocks"]) == set(FEATURE_GROUPS)
    for name, (start, end) in FEATURE_GROUPS.items():
        assert health["blocks"][name]["dims"] == end - start


def test_warming_up_before_enough_samples():
    health = _callback_with([np.zeros(PPO_OBS_SIZE)])._compute_obs_health()
    assert health["status"] == "warming_up"


# ─────────────────────────────────────────────────────────────
# Dashboard side
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "status,expect_banner",
    [("blind", True), ("bad", True), ("ok", False), ("good", False)],
)
def test_dashboard_raises_a_banner_only_for_real_faults(status, expect_banner):
    from dashboard.server import MetricsReader

    reader = MetricsReader.__new__(MetricsReader)
    out = reader._process_observation({
        "status": status,
        "alert": "x",
        "schema_version": PPO_OBS_VERSION,
        "schema_size": PPO_OBS_SIZE,
        "schema_hash": "abc123def456",
        "dead_dims": 0,
        "nan_count": 0,
        "mean_abs": 0.4,
        "samples": 512,
        "blocks": {},
    })
    assert out["status"] == status
    assert out["banner"] is expect_banner


def test_dashboard_flags_absent_health_rather_than_assuming_ok():
    """Silence must not read as health - that is how the outage stayed hidden."""
    from dashboard.server import MetricsReader

    reader = MetricsReader.__new__(MetricsReader)
    out = reader._process_observation(None)
    assert out["status"] == "unknown"
    assert out["banner"] is False
    assert "No observation health" in out["alert"]
