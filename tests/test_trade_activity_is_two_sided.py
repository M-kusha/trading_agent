"""The promotion gate must not pay for over-trading.

compute_composite_score scored trade activity as

    min(mean_trade_count / min_trade_count_avg, 2.0) / 2.0

which is monotonically increasing in trade count and saturates at 1.0. A policy
trading 366 times per episode against a stage target of 12 therefore earned a
perfect activity score - identical to one trading 24 times - and contributed a
full 1.0 to a composite of 0.94 with promotion_ready True.

That is the promotion gate rewarding exactly the behaviour
activity_consistency_penalty charges for. Observed in a live run at stage 0:
mean_trades 366.1, expected 12.0, component_scores["trade_activity"] = 1.0.
"""

from __future__ import annotations

import pytest

from envs.curriculum.config.thresholds import CompetenceThresholds, CompositeScoringConfig
from envs.curriculum.curriculum_config import get_all_stage_configs
from envs.curriculum.metrics import RollingStats, compute_composite_score


def _score(trade_count: float, thresholds: CompetenceThresholds) -> float:
    stats = RollingStats(window_size=50)
    stats.mean_trade_count = trade_count
    return compute_composite_score(
        stats, thresholds, CompositeScoringConfig()
    ).component_scores["trade_activity"]


def _thresholds(min_count: float = 5.0, max_count: float = 36.0) -> CompetenceThresholds:
    return CompetenceThresholds(min_trade_count_avg=min_count, max_trade_count_avg=max_count)


def test_the_observed_runaway_no_longer_scores_perfect():
    """366 trades/episode against a 12-trade target was scoring 1.0."""
    assert _score(366.0, _thresholds()) == pytest.approx(0.0)


def test_the_target_band_still_scores_well():
    th = _thresholds()
    assert _score(12.0, th) == pytest.approx(1.0)
    assert _score(24.0, th) == pytest.approx(1.0)


def test_under_trading_is_neutral_in_score_and_evidence_is_gated_separately():
    """Abstention is not a loss; promotion has a separate evidence check."""
    th = _thresholds()
    assert _score(0.0, th) == pytest.approx(1.0)
    assert _score(1.0, th) == pytest.approx(1.0)
    assert _score(2.0, th) == pytest.approx(_score(5.0, th))


def test_the_score_decays_monotonically_above_the_ceiling():
    th = _thresholds()
    counts = [36.0, 50.0, 70.0, 100.0, 150.0]
    scores = [_score(c, th) for c in counts]
    assert scores == sorted(scores, reverse=True), scores
    assert scores[0] > scores[-1]


def test_no_ceiling_configured_keeps_the_old_behaviour():
    """max_trade_count_avg = 0.0 must disable the ceiling, not zero the score."""
    th = CompetenceThresholds(min_trade_count_avg=5.0, max_trade_count_avg=0.0)
    assert _score(366.0, th) == pytest.approx(1.0)


@pytest.mark.parametrize("idx,stage", list(enumerate(get_all_stage_configs().values())))
def test_every_stage_defines_a_ceiling_above_its_own_target(idx, stage):
    """A ceiling below the stage's own expected count would block promotion."""
    c = stage.competence
    assert c.max_trade_count_avg > 0.0, f"stage {idx} has no trade ceiling"

    expected = stage.rewards.target_trades_per_1k_steps / 1000.0 * stage.max_steps_per_episode
    assert c.max_trade_count_avg > expected, (
        f"stage {idx} ceiling {c.max_trade_count_avg} is below its own expected "
        f"{expected:.1f} trades/episode"
    )
    assert c.max_trade_count_avg > c.min_trade_count_avg, f"stage {idx} band is inverted"
