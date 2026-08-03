"""A competence gate must be satisfiable by some policy.

is_fomo_entry is `setup_quality < setup_quality_threshold`, and setup_quality is
a property of market state rather than of the action - the agent chooses when to
enter, not what the setup is worth. Measured across 6,000 bars it spans
0.314-0.704 with p90 at 0.570, so a threshold of 0.70 sat at the very ceiling of
the range and classified essentially every entry as FOMO.

Stage 5 then demanded a FOMO rate at or below 0.20, i.e. that 80% of entries
clear a bar almost no bar clears. A live run sat at 0.9302 against that 0.20
requirement and could not promote past stage 5 no matter how well it traded.
The ladder also stepped straight from 1.0 at stage 4 to 0.20 at stage 5, so
there was no intermediate rung to climb.

These tests pin the threshold inside the metric's measured range and keep the
ladder continuous.
"""

from __future__ import annotations

import pytest

from envs.curriculum.curriculum_config import get_all_stage_configs

STAGES = list(get_all_stage_configs().values())

# Measured over 6,000 sampled bars with a random policy.
SETUP_QUALITY_MAX = 0.704
SETUP_QUALITY_P90 = 0.570
SETUP_QUALITY_P50 = 0.469


@pytest.mark.parametrize("idx", range(len(STAGES)))
def test_the_threshold_sits_inside_the_achievable_range(idx):
    thr = STAGES[idx].rewards.setup_quality_threshold
    assert thr < SETUP_QUALITY_MAX, (
        f"stage {idx}: setup_quality_threshold {thr} is at or above the measured "
        f"ceiling {SETUP_QUALITY_MAX} - every entry would be classed FOMO"
    )
    assert thr > SETUP_QUALITY_P50, (
        f"stage {idx}: threshold {thr} is below the median setup quality, so it "
        f"asks nothing of the policy"
    )


def test_a_selective_policy_can_clear_the_strictest_gate():
    """The tightest stage must be reachable by entering on good setups only."""
    strictest = min(s.competence.max_fomo_trade_rate for s in STAGES)
    thr = STAGES[-1].rewards.setup_quality_threshold

    assert thr <= SETUP_QUALITY_P90, (
        f"final threshold {thr} is above p90 {SETUP_QUALITY_P90}: clearing "
        f"{(1 - strictest) * 100:.0f}% of entries over it would require trading "
        f"almost exclusively at the distribution's extreme"
    )


def test_the_fomo_ladder_has_no_unreachable_cliff():
    rates = [s.competence.max_fomo_trade_rate for s in STAGES]

    assert rates == sorted(rates, reverse=True), f"FOMO ladder is not monotone: {rates}"

    for i in range(1, len(rates)):
        drop = rates[i - 1] - rates[i]
        assert drop <= 0.40, (
            f"stage {i - 1} -> {i}: FOMO gate tightens by {drop:.2f} in one step "
            f"({rates[i - 1]} -> {rates[i]}) - that is the cliff that stalled "
            f"promotion, not a rung"
        )


def test_the_gate_still_demands_selectivity_by_the_end():
    """Reachable must not mean trivial."""
    assert STAGES[-1].competence.max_fomo_trade_rate < 0.5, (
        "the final FOMO gate no longer asks the agent to be selective"
    )
    assert STAGES[0].competence.max_fomo_trade_rate >= 0.9, (
        "the first stage should not gate on FOMO at all - it is still exploring"
    )
