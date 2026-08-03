"""No competence gate may demand a value its metric cannot produce.

Four gates were found demanding the impossible, each stalling the curriculum
silently - training simply never promoted, with no error and no failed test:

  fomo_trade_rate     threshold sat at setup_quality's ceiling, so every entry
                      counted as FOMO and the <=0.20 gate could never be met
  mask_collapse_rate  numerator counted per tracking call, denominator per env
                      step, so the "rate" read 1.13 against a 1.0 limit
  stop_mode_rate      same mis-scaling
  avg_setup_quality   stages 8 and 9 asked for averages of 0.75 and 0.78 when
                      the metric's maximum over 5,000 sampled bars is 0.704

They share one shape: a threshold chosen without reference to the range of the
thing it gates. These bounds come from measurement, not from intuition.

Re-measure with scratchpad probes if the underlying signals change.
"""

from __future__ import annotations

import pytest

from envs.curriculum.curriculum_config import get_all_stage_configs

STAGES = list(get_all_stage_configs().values())

# Measured over 5,000-6,000 sampled bars with a random policy.
CEILINGS = {
    "min_avg_setup_quality": ("setup_quality", 0.704, 0.641),
    "min_entry_certainty_avg": ("entry_certainty", 0.736, 0.702),
    "setup_quality_threshold": ("setup_quality", 0.704, 0.641),
}

# A gate averaging above the p99 of what the market offers is not a standard,
# it is an impossibility with extra steps.
MAX_FRACTION_OF_CEILING = 0.96


@pytest.mark.parametrize("field,idx", [(f, i) for f in CEILINGS for i in range(len(STAGES))])
def test_gate_sits_below_its_metrics_ceiling(field, idx):
    metric, ceiling, _p99 = CEILINGS[field]
    stage = STAGES[idx]
    value = getattr(stage.competence, field, None)
    if value is None:
        value = getattr(stage.rewards, field, None)
    if value is None or value <= 0.0:
        return  # gate not active at this stage

    assert value < ceiling, (
        f"stage {idx}: {field}={value} is at or above the measured ceiling of "
        f"{metric} ({ceiling}). No policy can reach it - the curriculum would "
        f"stall here with no error reported."
    )
    assert value <= ceiling * MAX_FRACTION_OF_CEILING, (
        f"stage {idx}: {field}={value} is {value / ceiling:.0%} of {metric}'s "
        f"ceiling {ceiling}, which demands an average at the extreme of the "
        f"distribution"
    )


def test_a_rate_gate_never_exceeds_one():
    """Rates are bounded by definition; a limit above 1.0 gates nothing."""
    for idx, s in enumerate(STAGES):
        for field in ("max_fomo_trade_rate", "max_revenge_trade_rate",
                      "max_mask_collapse_rate", "max_stop_mode_rate",
                      "max_consecutive_loss_rate", "max_dd_breach_rate"):
            v = getattr(s.competence, field, None)
            if v is None:
                continue
            assert 0.0 <= v <= 1.0, f"stage {idx}: {field}={v} is outside [0,1]"


def test_quality_gates_tighten_without_becoming_unreachable():
    sq = [s.competence.min_avg_setup_quality for s in STAGES]
    active = [v for v in sq if v > 0]
    assert active == sorted(active), f"setup-quality ladder is not monotone: {sq}"
    assert max(active) < CEILINGS["min_avg_setup_quality"][1]


def test_the_fomo_threshold_leaves_headroom_above_the_median():
    """It must exclude poor setups without excluding everything."""
    _metric, ceiling, p99 = CEILINGS["setup_quality_threshold"]
    for idx, s in enumerate(STAGES):
        thr = s.rewards.setup_quality_threshold
        assert thr < p99, (
            f"stage {idx}: a FOMO threshold of {thr} is at or above p99 {p99}, "
            f"so virtually every entry is classed FOMO"
        )
