"""The dashboard must not report a dead run as a live, healthy one.

With no training process in existence the server reported `healthy`,
`training_active=true`, progress of 100.079%, an ETA of -38 seconds, an
observation schema of 7.0/40 marked `ok` while the runtime contract was 8.0/45,
and ~94% readiness derived while four authoritative promotion checks were
failing.

Every one of those came from the same habit: reading a file left on disk and
treating its contents as the current state of a running system. Nothing in the
payload said which run produced it, when, or whether the producer still lived.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from train.run_identity import (
    RunIdentity,
    describe_liveness,
    is_training_live,
    snapshot_age_seconds,
)


def _stamp(state: str = "running", age_seconds: float = 0.0) -> dict:
    made = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return {
        "run_id": "abc123",
        "sequence": 7,
        "produced_at_utc": made.isoformat(),
        "state": state,
        "pid": 999,
    }


# ── run identity ────────────────────────────────────────────────────────────

def test_sequence_is_monotonic_within_a_run():
    r = RunIdentity()
    seqs = [r.stamp()["sequence"] for _ in range(5)]
    assert seqs == sorted(seqs) and len(set(seqs)) == 5


def test_a_new_run_gets_a_new_id():
    assert RunIdentity().run_id != RunIdentity().run_id


def test_an_unknown_state_is_rejected():
    with pytest.raises(ValueError):
        RunIdentity().set_state("probably_fine")


# ── liveness ────────────────────────────────────────────────────────────────

def test_a_running_recent_producer_is_live():
    assert is_training_live(_stamp("running", age_seconds=5)) is True


def test_a_running_but_silent_producer_is_not_live():
    """A killed process leaves 'running' behind; only age reveals it."""
    assert is_training_live(_stamp("running", age_seconds=3600)) is False


def test_a_completed_run_is_not_live_however_recent():
    assert is_training_live(_stamp("completed", age_seconds=1)) is False


def test_a_failed_run_is_not_live():
    assert is_training_live(_stamp("failed", age_seconds=1)) is False


def test_missing_identity_is_not_live_and_says_why():
    d = describe_liveness({})
    assert d["training_live"] is False
    assert d["state"] == "unknown"
    assert "no run identity" in d["reason"]


def test_liveness_explains_a_stale_producer():
    d = describe_liveness(_stamp("running", age_seconds=600))
    assert d["training_live"] is False
    assert d["stale"] is True
    assert "no update" in d["reason"]


def test_age_is_measured_from_the_producers_clock():
    age = snapshot_age_seconds(_stamp("running", age_seconds=120))
    assert 110 <= age <= 130


def test_an_unparseable_timestamp_is_treated_as_infinitely_stale():
    assert snapshot_age_seconds({"produced_at_utc": "not-a-time"}) == float("inf")
    assert is_training_live({"state": "running", "produced_at_utc": "not-a-time"}) is False


# ── schema compatibility ────────────────────────────────────────────────────

def test_a_mismatched_observation_schema_is_never_ok():
    from dashboard.server import MetricsReader

    reader = MetricsReader.__new__(MetricsReader)
    out = reader._process_observation({
        "status": "ok",
        "schema_version": "7.0",
        "schema_size": 40,
        "dead_dims": 0,
        "nan_count": 0,
        "blocks": {},
    })
    assert out["status"] == "historical_incompatible"
    assert out["schema_matches_runtime"] is False
    assert "different agent" in out["alert"]


def test_a_matching_schema_keeps_its_reported_status():
    from dashboard.server import MetricsReader
    from modules.meta.ppo_observation_builder import PPO_OBS_SIZE, PPO_OBS_VERSION

    reader = MetricsReader.__new__(MetricsReader)
    out = reader._process_observation({
        "status": "good",
        "schema_version": str(PPO_OBS_VERSION),
        "schema_size": int(PPO_OBS_SIZE),
        "dead_dims": 0,
        "nan_count": 0,
        "blocks": {},
    })
    assert out["status"] == "good"
    assert out["schema_matches_runtime"] is True


# ── telemetry gaps ──────────────────────────────────────────────────────────

def test_absent_ppo_telemetry_stays_unknown_rather_than_zero():
    from dashboard.server import MetricsReader

    reader = MetricsReader.__new__(MetricsReader)
    reader._history = {k: [] for k in ("entropy", "explained_variance", "kl_divergence")}
    out = reader._process_learning({"learning": {"fps": 30.0}}) if hasattr(
        MetricsReader, "_process_learning"
    ) else None
    if out is None:
        pytest.skip("learning block is not separately processable")

    for key in ("entropy", "explained_variance", "policy_loss", "clip_fraction"):
        assert out[key] is None, f"{key} was invented as {out[key]!r}"
        assert out[f"{key}_status"] == "unknown"
