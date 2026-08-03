"""An evaluation is only evidence about the model it names.

Nothing previously stopped the evaluator loading a different dataset, a
different split or a later cutoff than the run that produced the checkpoint, and
reporting the result as though it described that model.

The concrete failure was `last_trained`. It was derived from the loaded CSV's
maximum timestamp, so the moment broker bars were appended it jumped to the
newest bar, `load_ftmo_data` found nothing after it, and the three-way
comparison silently collapsed to two - with no error and no missing-section
warning. The model's own record of where training ended is the only correct
source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from evaluate.holdout_report import (
    assert_no_training_overlap,
    bind_evaluation_to_provenance,
    training_end_from_provenance,
)


def _provenance(**over):
    p = {
        "datasets": {
            "loaded": {"fingerprint": "aaa", "boundary": "2026-08-03T08:15:00Z"},
            "train": {"fingerprint": "bbb", "boundary": "2026-06-12T23:45:00Z"},
            "holdout": {"fingerprint": "ccc", "boundary": "2026-06-15T01:00:00Z"},
        },
        "arguments": {
            "resolved_holdout_split_at": "2026-06-15",
            "data_cutoff": "2026-08-03T08:15:00Z",
        },
    }
    p.update(over)
    return p


def _manifests(loaded="aaa", train="bbb", holdout="ccc"):
    return (
        {"dataset_fingerprint": loaded},
        {"dataset_fingerprint": train},
        {"dataset_fingerprint": holdout},
    )


# ── training end ────────────────────────────────────────────────────────────

def test_training_end_comes_from_the_model_not_the_csv():
    assert training_end_from_provenance(_provenance()) == "2026-06-12T23:45:00Z"


def test_training_end_falls_back_to_frame_timestamps():
    prov = {"datasets": {"train": {"frames": [
        {"last_time": "2026-06-01T00:00:00Z"},
        {"last_time": "2026-06-12T23:45:00Z"},
    ]}}}
    assert training_end_from_provenance(prov) == "2026-06-12T23:45:00Z"


def test_training_end_is_none_when_unrecorded():
    assert training_end_from_provenance({}) is None


# ── binding ─────────────────────────────────────────────────────────────────

def test_matching_provenance_binds():
    loaded, train, holdout = _manifests()
    out = bind_evaluation_to_provenance(
        _provenance(),
        loaded_manifest=loaded, train_manifest=train, holdout_manifest=holdout,
        resolved_split_at="2026-06-15", data_cutoff="2026-08-03T08:15:00Z",
        allow_unbound=False,
    )
    assert out["bound"] is True
    assert out["mismatches"] == []


@pytest.mark.parametrize("role,swapped", [
    ("loaded", {"loaded": "different"}),
    ("train", {"train": "different"}),
    ("holdout", {"holdout": "different"}),
])
def test_a_different_dataset_is_refused(role, swapped):
    loaded, train, holdout = _manifests(**swapped)
    with pytest.raises(ValueError, match="not bound"):
        bind_evaluation_to_provenance(
            _provenance(),
            loaded_manifest=loaded, train_manifest=train, holdout_manifest=holdout,
            resolved_split_at="2026-06-15", data_cutoff="2026-08-03T08:15:00Z",
            allow_unbound=False,
        )


def test_a_different_split_is_refused():
    loaded, train, holdout = _manifests()
    with pytest.raises(ValueError, match="holdout split differs"):
        bind_evaluation_to_provenance(
            _provenance(),
            loaded_manifest=loaded, train_manifest=train, holdout_manifest=holdout,
            resolved_split_at="2026-01-01", data_cutoff="2026-08-03T08:15:00Z",
            allow_unbound=False,
        )


def test_a_different_cutoff_is_refused():
    loaded, train, holdout = _manifests()
    with pytest.raises(ValueError, match="data cutoff differs"):
        bind_evaluation_to_provenance(
            _provenance(),
            loaded_manifest=loaded, train_manifest=train, holdout_manifest=holdout,
            resolved_split_at="2026-06-15", data_cutoff="2027-01-01T00:00:00Z",
            allow_unbound=False,
        )


def test_mismatches_are_reported_when_explicitly_allowed():
    loaded, train, holdout = _manifests(train="different")
    out = bind_evaluation_to_provenance(
        _provenance(),
        loaded_manifest=loaded, train_manifest=train, holdout_manifest=holdout,
        resolved_split_at="2026-06-15", data_cutoff="2026-08-03T08:15:00Z",
        allow_unbound=True,
    )
    assert out["bound"] is False
    assert out["mismatches"], "a mismatch must still be recorded, not swallowed"


def test_an_unbound_diagnostic_does_not_pretend_to_be_bound():
    out = bind_evaluation_to_provenance(
        {"status": "UNBOUND_MODEL_ALLOWED_FOR_DIAGNOSTICS"},
        loaded_manifest={}, train_manifest={}, holdout_manifest={},
        resolved_split_at="x", data_cutoff="y", allow_unbound=True,
    )
    assert out["bound"] is False


# ── overlap ─────────────────────────────────────────────────────────────────

def _dataset(first: str, last: str):
    times = pd.date_range(first, last, freq="15min", tz="UTC")
    return {"XAUUSD": {"M15": pd.DataFrame({"time": times, "close": 1.0})}}


def test_evaluation_bars_at_or_before_training_end_are_refused():
    data = _dataset("2026-06-01", "2026-07-01")
    with pytest.raises(ValueError, match="in-sample"):
        assert_no_training_overlap(data, "XAUUSD", "2026-06-12T23:45:00Z")


def test_strictly_later_bars_are_accepted():
    data = _dataset("2026-06-15", "2026-07-01")
    assert_no_training_overlap(data, "XAUUSD", "2026-06-12T23:45:00Z")


def test_the_boundary_bar_itself_counts_as_in_sample():
    """Equality must be in-sample - the model saw that bar."""
    data = _dataset("2026-06-13", "2026-07-01")
    with pytest.raises(ValueError):
        assert_no_training_overlap(data, "XAUUSD", "2026-06-13T00:00:00Z")


def test_no_training_end_means_no_claim_and_no_crash():
    data = _dataset("2026-06-15", "2026-07-01")
    assert_no_training_overlap(data, "XAUUSD", None)
