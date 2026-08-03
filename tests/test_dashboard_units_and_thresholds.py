"""Units belong in the name, and gates belong to the curriculum.

Two defects, both of which made the panel confidently wrong.

`mean_win_rate` was a fraction at the top level of the payload and a percent
inside the `trading` block - in the same file - so the consumer guessed from
magnitude with `if 0 < x < 1: x *= 100`. That rule is undecidable by
construction: a genuine 0.8% drawdown and a catastrophic 80% one are the same
number under it.

The KPI thresholds were a copy of the curriculum's gates pasted into the
frontend, and the copy drifted. It graded against a 53% win rate and a 1.28
profit factor after the ladder had moved to 40% and 1.40, so the dashboard was
judging the agent by requirements that no longer existed.
"""

from __future__ import annotations

import pytest

from dashboard.server import MetricsReader


def _reader() -> MetricsReader:
    r = MetricsReader.__new__(MetricsReader)
    r._history = {}
    return r


# ── explicit units ──────────────────────────────────────────────────────────

def test_explicit_percent_keys_are_preferred():
    r = _reader()
    got = r._percent(
        {"win_rate_pct": 61.5, "mean_win_rate": 0.42},
        {},
        "win_rate_pct",
        "mean_win_rate",
        legacy_is_fraction=False,
    )
    assert got == pytest.approx(61.5)


def test_a_legacy_fraction_is_converted_by_convention_not_magnitude():
    r = _reader()
    # 0.008 is a genuine 0.8% drawdown. Magnitude guessing cannot tell it from
    # 0.8 meaning 80%; the recorded convention can.
    assert r._percent({}, {"max_drawdown": 0.008}, "drawdown_pct",
                      "max_drawdown", legacy_is_fraction=True) == pytest.approx(0.8)


def test_a_small_but_real_drawdown_is_not_inflated():
    r = _reader()
    dd = r._normalize_drawdown_percent({"max_drawdown": 0.004}, {})
    assert dd == pytest.approx(0.4), f"0.4% drawdown was reported as {dd}%"


def test_an_explicit_drawdown_percent_wins_over_the_legacy_key():
    r = _reader()
    dd = r._normalize_drawdown_percent({"max_drawdown": 0.02}, {"drawdown_pct": 3.5})
    assert dd == pytest.approx(3.5)


def test_a_missing_value_is_zero_not_a_guess():
    r = _reader()
    assert r._percent({}, {}, "drawdown_pct", "max_drawdown", legacy_is_fraction=True) == 0.0


def test_both_callbacks_emit_the_same_explicit_keys():
    import inspect

    from train.callbacks.curriculum_callback import CurriculumTrainingCallback
    from train.callbacks.episode_callback import VecEpisodeTradingCallback

    for cls in (CurriculumTrainingCallback, VecEpisodeTradingCallback):
        src = inspect.getsource(cls)
        assert "win_rate_pct" in src, f"{cls.__name__} does not name its win-rate unit"
        assert "drawdown_pct" in src, f"{cls.__name__} does not name its drawdown unit"


# ── effective stage gates ───────────────────────────────────────────────────

def test_requirements_come_from_the_curriculum():
    r = _reader()
    out = r._effective_stage_requirements(
        {"curriculum_stage": "LIVE_READY", "curriculum_progress": {"stage_index": 9}}
    )
    assert out["available"] is True
    assert out["competence"]["min_win_rate"] == pytest.approx(0.40)
    assert out["competence"]["min_profit_factor"] == pytest.approx(1.40)


def test_requirements_resolve_by_index_when_the_name_is_missing():
    r = _reader()
    out = r._effective_stage_requirements({"curriculum_progress": {"stage_index": 9}})
    assert out["available"] is True
    assert out["stage_index"] == 9


def test_an_unknown_stage_reports_unavailable_rather_than_permissive():
    r = _reader()
    out = r._effective_stage_requirements({"curriculum_stage": "NOT_A_STAGE"})
    assert out["available"] is False
    assert "no requirements" in out["reason"]


def test_the_frontend_prefers_runtime_gates_over_its_own_table():
    html = (__import__("pathlib").Path("dashboard/index.html")).read_text(encoding="utf-8")
    assert "function effectiveThresholds" in html
    assert "effectiveThresholds(data, phase)" in html, (
        "the KPI status still reads the static table directly"
    )
