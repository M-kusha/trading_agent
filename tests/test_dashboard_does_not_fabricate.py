"""A metric the dashboard never received must not render as a good number.

The observation outage stayed hidden because a missing panel looked like a
healthy one. The same shape appeared again in the quality panel: the curriculum
callback's "quality" block carried only mean_profit_factor, mean_r_multiple and
mean_entry_quality, while the dashboard read six keys with
`q.get("max_consecutive_losses", 0)`. In curriculum mode - the mode that
matters - the panel therefore reported a maximum losing streak of 0 across
thousands of trades at a ~50% win rate. That is not an achievable result; it
was the default value of a key nobody wrote.

These tests pin both halves: the callback must emit the fields, and the
dashboard must distinguish absent from zero.
"""

from __future__ import annotations

import pytest

from dashboard.server import MetricsReader


def test_absent_streak_data_is_not_reported_as_zero():
    """The exact payload the curriculum callback used to write."""
    reader = MetricsReader.__new__(MetricsReader)
    reader._history = {}
    out = reader._process_quality({
        "quality": {
            "mean_profit_factor": 1.05,
            "mean_r_multiple": 0.004,
            "mean_entry_quality": 0.49,
        }
    })

    assert out["streak_data_available"] is False
    assert out["max_consecutive_losses"] is None, (
        "a streak that was never measured must not render as 0 - that reads as "
        "a perfect run"
    )
    assert out["avg_consecutive_losses"] is None
    assert out["consecutive_loss_streak_rate"] is None


def test_present_streak_data_is_passed_through():
    reader = MetricsReader.__new__(MetricsReader)
    reader._history = {}
    out = reader._process_quality({
        "quality": {
            "mean_profit_factor": 1.05,
            "mean_r_multiple": 0.004,
            "mean_entry_quality": 0.49,
            "max_consecutive_losses": 7,
            "avg_consecutive_losses": 2.5,
            "consecutive_loss_streak_rate": 0.4,
        }
    })

    assert out["streak_data_available"] is True
    assert out["max_consecutive_losses"] == 7
    assert out["avg_consecutive_losses"] == pytest.approx(2.5)
    assert out["consecutive_loss_streak_rate"] == pytest.approx(0.4)


def test_a_genuine_zero_streak_is_still_reported_as_zero():
    """Absent and zero must stay distinguishable in both directions."""
    reader = MetricsReader.__new__(MetricsReader)
    reader._history = {}
    out = reader._process_quality({
        "quality": {
            "mean_profit_factor": 1.0,
            "mean_r_multiple": 0.0,
            "mean_entry_quality": 0.5,
            "max_consecutive_losses": 0,
            "avg_consecutive_losses": 0.0,
            "consecutive_loss_streak_rate": 0.0,
        }
    })

    assert out["streak_data_available"] is True
    assert out["max_consecutive_losses"] == 0


def test_both_callbacks_emit_the_same_quality_keys():
    """Whichever callback writes last, the panel must mean the same thing."""
    import inspect

    from train.callbacks.curriculum_callback import CurriculumTrainingCallback

    src = inspect.getsource(CurriculumTrainingCallback._consecutive_loss_stats)
    for key in (
        "max_consecutive_losses",
        "avg_consecutive_losses",
        "consecutive_loss_streak_rate",
    ):
        assert key in src, f"curriculum callback stopped emitting {key}"


def test_curriculum_callback_reports_a_measured_streak():
    from collections import deque

    from train.callbacks.curriculum_callback import CurriculumTrainingCallback

    cb = CurriculumTrainingCallback.__new__(CurriculumTrainingCallback)
    cb._ep_consecutive_losses = deque([1, 4, 2, 0, 3])

    stats = cb._consecutive_loss_stats()
    assert stats["max_consecutive_losses"] == 4
    assert stats["avg_consecutive_losses"] == pytest.approx(2.0)
    # two of five episodes reached a streak of 3 or more
    assert stats["consecutive_loss_streak_rate"] == pytest.approx(0.4)


def test_curriculum_callback_reports_none_before_any_episode():
    from collections import deque

    from train.callbacks.curriculum_callback import CurriculumTrainingCallback

    cb = CurriculumTrainingCallback.__new__(CurriculumTrainingCallback)
    cb._ep_consecutive_losses = deque()

    stats = cb._consecutive_loss_stats()
    assert stats["max_consecutive_losses"] is None
