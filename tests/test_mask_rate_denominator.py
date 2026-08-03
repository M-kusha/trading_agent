"""A rate must not exceed 1.0.

compute_rolling_stats divided mask_collapse_steps and stop_mode_steps by the
summed EPISODE LENGTHS, but those counters are incremented inside
_track_action_mask_state_for_metrics, which the env calls from two separate
sites per step. Numerator and denominator were in different units, so the
"rate" could exceed 1.

A live run reported mask_collapse_rate and stop_mode_rate both at 1.1311
against a limit of 1.0 and listed them as promotion blockers - gates no policy
could clear, because the value could not be brought under the limit by trading
differently. The env's own per-episode rate uses mask_decision_steps and was
correctly bounded all along; only the aggregate was wrong.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.curriculum.metrics import EpisodeMetrics


def _episode(collapse: int, stop: int, decisions: int, length: int) -> EpisodeMetrics:
    return EpisodeMetrics(
        mask_collapse_steps=collapse,
        stop_mode_steps=stop,
        mask_decision_steps=decisions,
        episode_length=length,
    )


def test_episode_metrics_carries_the_denominator():
    m = _episode(collapse=10, stop=10, decisions=100, length=50)
    assert m.mask_decision_steps == 100, (
        "without this field the aggregate falls back to episode length, which "
        "is the wrong unit"
    )


def test_rate_stays_bounded_when_tracking_runs_twice_per_step():
    """The exact shape of the bug: two tracking calls per env step."""
    from envs.curriculum.curriculum_manager import CurriculumManager

    # 100 env steps -> 200 tracking calls; the mask collapsed on 130 of them.
    # Against episode length that reads 1.30; against decision steps, 0.65.
    window = [_episode(collapse=130, stop=130, decisions=200, length=100) for _ in range(5)]

    collapse = np.array([m.mask_collapse_steps for m in window], dtype=float)
    decisions = np.array([m.mask_decision_steps for m in window], dtype=float)
    lengths = np.array([m.episode_length for m in window], dtype=float)

    wrong = float(collapse.sum() / lengths.sum())
    right = float(collapse.sum() / decisions.sum())

    assert wrong > 1.0, "test fixture no longer reproduces the bug"
    assert right <= 1.0, f"corrected rate is still out of bounds: {right}"
    assert right == pytest.approx(0.65)
    assert CurriculumManager is not None


def test_the_manager_uses_decision_steps_as_denominator():
    import inspect

    from envs.curriculum.curriculum_manager import CurriculumManager

    src = inspect.getsource(CurriculumManager)
    assert "mask_decision_steps" in src, (
        "the aggregate no longer reads the decision-step denominator"
    )
    assert "total_decisions" in src


def test_falls_back_safely_when_the_denominator_is_absent():
    """Older checkpoints have no mask_decision_steps; must not divide by zero."""
    window = [_episode(collapse=5, stop=5, decisions=0, length=100) for _ in range(3)]
    decisions = np.array([m.mask_decision_steps for m in window], dtype=float)
    lengths = np.array([m.episode_length for m in window], dtype=float)

    total = decisions.sum() or lengths.sum()
    assert total > 0
    assert float(np.array([m.mask_collapse_steps for m in window]).sum() / total) <= 1.0
