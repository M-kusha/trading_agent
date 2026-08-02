"""The competence ladder must be internally consistent with a 2:1 target.

The old ladder demanded a 54% win rate at profit factor 1.32, which describes a
roughly 1:1 system. A 2:1 reward:risk target breaks even at a 33.3% win rate,
so demanding 54% is not "more ambitious" - it is a different and mutually
exclusive strategy. The two cannot be pursued at once.

It also shrank the optimal hold from 12 bars to 6 (1.5h) while requiring larger
R, and activated the trailing stop at ~0.2R with a 30% retrace, so a winner
reaching 1R was stopped out at 0.7R. Measured on this data at a 1.5xATR stop,
P(reach 2R before the stop) is 0.244 at a 24-bar horizon, 0.276 at 48 and 0.307
at 96 - a 2R target needs room and time, not a faster exit.
"""

from __future__ import annotations

import pytest

from envs.curriculum.curriculum_config import get_all_stage_configs

TARGET_R = 2.0
STAGES = list(get_all_stage_configs().values())


def _breakeven_win_rate(r: float) -> float:
    return 1.0 / (1.0 + r)


def test_the_final_win_rate_gate_clears_breakeven_for_the_target():
    final = STAGES[-1].competence.min_win_rate
    breakeven = _breakeven_win_rate(TARGET_R)

    assert final > breakeven, (
        f"final win-rate gate {final:.2f} is at or below the {TARGET_R}:1 "
        f"breakeven of {breakeven:.3f} - the ladder cannot certify profitability"
    )
    assert final < 0.50, (
        f"final win-rate gate {final:.2f} implies a ~1:1 system; a {TARGET_R}:1 "
        f"strategy does not also win half its trades"
    )


def test_the_final_gates_are_mutually_consistent():
    """win rate, profit factor and expectancy must describe one strategy."""
    c = STAGES[-1].competence
    wr = c.min_win_rate

    implied_pf = (wr * TARGET_R) / max((1.0 - wr) * 1.0, 1e-9)
    implied_expectancy = wr * TARGET_R - (1.0 - wr)

    assert c.min_profit_factor == pytest.approx(implied_pf, abs=0.15), (
        f"profit-factor gate {c.min_profit_factor} does not match the "
        f"{implied_pf:.2f} implied by a {wr:.0%} win rate at {TARGET_R}:1"
    )
    assert c.min_avg_r_multiple == pytest.approx(implied_expectancy, abs=0.05), (
        f"R gate {c.min_avg_r_multiple} does not match the "
        f"{implied_expectancy:+.2f}R implied by a {wr:.0%} win rate at {TARGET_R}:1"
    )


def test_hold_time_grows_with_the_r_requirement():
    """Larger R needs more bars, so the optimal hold must not shrink."""
    optimal = [s.rewards.optimal_trade_bars for s in STAGES]
    assert optimal == sorted(optimal), f"optimal hold shrinks across stages: {optimal}"
    assert optimal[-1] >= 40, (
        f"final optimal hold {optimal[-1]} bars ({optimal[-1] * 0.25:.1f}h) is too "
        f"short to reach {TARGET_R}R"
    )


def test_the_bonus_window_does_not_cut_the_target_short():
    for i, s in enumerate(STAGES):
        r = s.rewards
        assert r.max_trade_bars_for_bonus > r.optimal_trade_bars, (
            f"stage {i}: bonus window {r.max_trade_bars_for_bonus} does not exceed "
            f"the optimal hold {r.optimal_trade_bars}"
        )
    assert STAGES[-1].rewards.max_trade_bars_for_bonus >= 96


def test_the_trailing_stop_activates_above_1r():
    """Activating at 0.2R with a 30% retrace caps winners near 1R."""
    for i, s in enumerate(STAGES):
        ov = s.env_overrides or {}
        stop = float(ov.get("hard_stop_loss_eur", 220.0))
        activation = float(ov.get("trailing_activation_eur", 100.0))
        in_r = activation / max(stop, 1e-9)
        assert in_r >= 1.0, (
            f"stage {i}: trailing activates at {in_r:.2f}R, which stops winners "
            f"before they can reach {TARGET_R}R"
        )


def test_holding_cost_does_not_outweigh_the_target():
    """Time cost over a full hold must stay small against the R being sought."""
    for i, s in enumerate(STAGES):
        r = s.rewards
        total = r.holding_cost_per_bar * r.max_trade_bars_for_bonus
        assert total <= 0.05, (
            f"stage {i}: holding cost totals {total:.4f} over "
            f"{r.max_trade_bars_for_bonus} bars, which fights the long holds "
            f"a {TARGET_R}R target needs"
        )


@pytest.mark.parametrize("field,rising", [
    ("min_win_rate", True),
    ("min_profit_factor", True),
    ("min_avg_r_multiple", True),
])
def test_gates_still_tighten_across_the_curriculum(field, rising):
    values = [getattr(s.competence, field) for s in STAGES]
    assert values == sorted(values) if rising else values == sorted(values, reverse=True), (
        f"{field} is not monotonic across stages: {values}"
    )
