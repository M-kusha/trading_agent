"""The evaluator must state its scope, admit its uncertainty, and fail closed.

Three defects in the published backtest motivated these, all of which produced
confident-looking numbers that were wrong:

  * `total_pnl` held a MEAN across independently reset accounts while reading
    as a total. Summed across ten €100k resets and divided by one €100k, a
    +2.57% mean was published as +25.70%.
  * A 1.96-sigma bound was computed from as few as two blocks and printed as a
    confidence interval. That is arithmetic, not inference.
  * The verdict compared the model to `one-shot-long` - a policy that trades
    through the environment's stops, spread and drawdown veto - and printed
    "BEATS buy-and-hold". It could return a confident conclusion from one
    window and three trades, against a benchmark that is not the market.
"""

from __future__ import annotations

import numpy as np
import pytest

from evaluate.holdout_report import (
    MIN_INDEPENDENT_BLOCKS,
    MIN_TRADES_FOR_VERDICT,
    Result,
    _block_bootstrap_lower_bound,
    _reconcile_episode,
    evaluate_section_verdict,
)


# ── scope naming ────────────────────────────────────────────────────────────

def test_mean_and_pooled_are_separate_fields():
    r = Result(name="x")
    assert hasattr(r, "mean_episode_pnl_eur")
    assert hasattr(r, "pooled_sample_pnl_eur")
    assert not hasattr(r, "total_pnl"), (
        "total_pnl held a mean while reading as a total - the name that caused "
        "a 10x published error must not come back"
    )


# ── reconciliation ──────────────────────────────────────────────────────────

def test_reconciliation_passes_on_consistent_books():
    out = _reconcile_episode(
        equity_change=100.0, trade_pnl=100.0,
        long_pnl=60.0, short_pnl=40.0,
        recorded_trades=5, reported_trades=5, position_open=False,
    )
    assert out["failures"] == []


def test_direction_split_must_sum_to_trade_pnl():
    out = _reconcile_episode(
        equity_change=100.0, trade_pnl=100.0,
        long_pnl=60.0, short_pnl=10.0,      # 70 != 100
        recorded_trades=5, reported_trades=5, position_open=False,
    )
    assert any("long+short" in f for f in out["failures"])


def test_trade_counts_must_agree():
    out = _reconcile_episode(
        equity_change=100.0, trade_pnl=100.0,
        long_pnl=60.0, short_pnl=40.0,
        recorded_trades=5, reported_trades=7, position_open=False,
    )
    assert any("recorded" in f for f in out["failures"])


def test_unexplained_equity_change_fails_when_flat():
    out = _reconcile_episode(
        equity_change=250.0, trade_pnl=100.0,
        long_pnl=60.0, short_pnl=40.0,
        recorded_trades=5, reported_trades=5, position_open=False,
    )
    assert any("residual" in f for f in out["failures"])


def test_an_open_position_is_the_one_tolerated_residual():
    out = _reconcile_episode(
        equity_change=250.0, trade_pnl=100.0,
        long_pnl=60.0, short_pnl=40.0,
        recorded_trades=5, reported_trades=5, position_open=True,
    )
    assert out["failures"] == []
    assert out["unrealised_residual_eur"] == pytest.approx(150.0)


# ── confidence interval ─────────────────────────────────────────────────────

def test_the_bootstrap_bound_sits_below_the_mean():
    returns = np.array([1.0, 2.0, -0.5, 3.0, 0.2, 1.5, -1.0, 2.2, 0.8, 1.1])
    low = _block_bootstrap_lower_bound(returns, seed=0)
    assert low < returns.mean()


def test_the_bound_is_deterministic_for_a_seed():
    returns = np.array([1.0, 2.0, -0.5, 3.0, 0.2, 1.5, -1.0, 2.2, 0.8, 1.1])
    assert _block_bootstrap_lower_bound(returns, seed=7) == pytest.approx(
        _block_bootstrap_lower_bound(returns, seed=7)
    )


def test_a_wider_spread_gives_a_lower_bound():
    tight = np.full(12, 1.0)
    wide = np.array([1.0, -8.0, 9.0, 1.0, -7.0, 8.0, 1.0, -6.0, 7.0, 1.0, -5.0, 6.0])
    assert _block_bootstrap_lower_bound(wide, seed=0) < _block_bootstrap_lower_bound(tight, seed=0)


def test_the_preregistered_minimum_is_not_trivial():
    assert MIN_INDEPENDENT_BLOCKS >= 10
    assert MIN_TRADES_FOR_VERDICT >= 30


# ── fail-closed verdict ─────────────────────────────────────────────────────

def _model_row(**over):
    row = {
        "name": "trained-model", "trades": 200, "episodes": 12,
        "reconciliation_checked": 12, "return_pct": 3.0,
        "return_ci_low_pct": 0.8, "passive_return_pct": 1.0,
        "max_drawdown_pct": 4.0,
    }
    row.update(over)
    return row


def _rows(model_over=None, flat_return=0.0):
    return [
        _model_row(**(model_over or {})),
        {"name": "always-flat", "return_pct": flat_return, "trades": 0},
    ]


_GOOD_META = {"independent_windows": 12, "overlapping_windows": False}


def test_a_clean_result_is_supported_but_only_for_development():
    v = evaluate_section_verdict(_rows(), _GOOD_META)
    assert v["verdict"] == "SUPPORTED_DEVELOPMENT_ONLY"
    assert any("not live approval" in r for r in v["reasons"])


def test_too_few_windows_blocks_any_conclusion():
    v = evaluate_section_verdict(_rows(), {"independent_windows": 1})
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "independent_windows" in v["blocking"]


def test_too_few_trades_blocks_any_conclusion():
    v = evaluate_section_verdict(_rows({"trades": 5}), _GOOD_META)
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "trades" in v["blocking"]


def test_overlapping_windows_block_any_conclusion():
    v = evaluate_section_verdict(_rows(), {"independent_windows": 12, "overlapping_windows": True})
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_unreconciled_episodes_block_any_conclusion():
    v = evaluate_section_verdict(_rows({"reconciliation_checked": 3}), _GOOD_META)
    assert v["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "reconciliation" in v["blocking"]


def test_a_lower_bound_at_or_below_zero_is_not_support():
    v = evaluate_section_verdict(_rows({"return_ci_low_pct": -0.2}), _GOOD_META)
    assert v["verdict"] == "NOT_SUPPORTED"
    assert "positive_lower_bound" in v["blocking"]


def test_failing_to_beat_standing_aside_is_not_support():
    v = evaluate_section_verdict(_rows({"return_pct": 0.5}, flat_return=1.0), _GOOD_META)
    assert v["verdict"] == "NOT_SUPPORTED"
    assert "beats_flat" in v["blocking"]


def test_failing_to_beat_the_passive_market_is_not_support():
    v = evaluate_section_verdict(
        _rows({"return_pct": 2.0, "passive_return_pct": 5.0}), _GOOD_META
    )
    assert v["verdict"] == "NOT_SUPPORTED"
    assert "beats_market" in v["blocking"]


def test_a_drawdown_breach_is_not_support():
    v = evaluate_section_verdict(_rows({"max_drawdown_pct": 14.0}), _GOOD_META)
    assert v["verdict"] == "NOT_SUPPORTED"
    assert "drawdown" in v["blocking"]


def test_a_missing_flat_control_blocks_support():
    v = evaluate_section_verdict([_model_row()], _GOOD_META)
    assert v["verdict"] == "NOT_SUPPORTED"
    assert "flat_control_missing" in v["blocking"]


def test_no_model_yields_no_verdict():
    v = evaluate_section_verdict([{"name": "always-flat", "return_pct": 0.0}], _GOOD_META)
    assert v["verdict"] == "NO_MODEL_EVALUATED"
