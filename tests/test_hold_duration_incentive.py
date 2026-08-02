"""The duration bonus must pay most at the target hold, not at zero bars.

time_efficiency paid scale * (1 - bars/optimal * 0.5), which is largest at zero
bars and falls to exactly half AT the optimum. With optimal=12 a 1-bar exit
collected 95.8% of the bonus and a 12-bar hold collected 50%. The reward was
teaching the scalping it was supposed to discourage.

It showed up exactly that way in training: median hold 1.0 bars across 46,844
trades, mean MFE 0.157R, and trailing_stop firing 15 times (0.03%) because no
trade ever survived long enough to reach the trailing activation.

Stages 0-1 additionally had every duration incentive zeroed - time efficiency
off, agent_close_bonus 0, trailing_stop_bonus 0 - so the habit formed there
with nothing opposing it and carried into later stages.
"""

from __future__ import annotations

import pytest

from envs.curriculum.curriculum_config import get_all_stage_configs

STAGES = list(get_all_stage_configs().values())


def _bonus(cfg, bars_held: float) -> float:
    """Mirror of the shaping in trade_reward for a winning trade."""
    optimal = max(cfg.optimal_trade_bars, 1)
    if bars_held <= optimal:
        factor = bars_held / optimal
    else:
        denom = max(cfg.max_trade_bars_for_bonus - optimal, 1)
        factor = 1.0 - (bars_held - optimal) / denom
    return cfg.time_efficiency_scale * max(0.0, min(1.0, factor))


@pytest.mark.parametrize("idx", range(len(STAGES)))
def test_holding_to_target_beats_exiting_instantly(idx):
    cfg = STAGES[idx].rewards
    assert _bonus(cfg, cfg.optimal_trade_bars) > _bonus(cfg, 1) * 3, (
        f"stage {idx}: a 1-bar exit is not clearly worse than holding to "
        f"{cfg.optimal_trade_bars} bars"
    )


@pytest.mark.parametrize("idx", range(len(STAGES)))
def test_the_bonus_peaks_at_the_optimal_hold(idx):
    cfg = STAGES[idx].rewards
    optimal = cfg.optimal_trade_bars
    peak = _bonus(cfg, optimal)

    for bars in (1, 2, optimal // 2, optimal * 2, cfg.max_trade_bars_for_bonus):
        assert _bonus(cfg, bars) <= peak + 1e-12, (
            f"stage {idx}: {bars} bars pays more than the optimum {optimal}"
        )


@pytest.mark.parametrize("idx", range(len(STAGES)))
def test_every_stage_has_a_live_duration_incentive(idx):
    """Stages 0-1 previously had all of these at zero."""
    r = STAGES[idx].rewards
    assert r.time_efficiency_enabled is True, f"stage {idx}: duration bonus disabled"
    assert r.time_efficiency_scale > 0.0, f"stage {idx}: duration bonus has zero scale"
    assert r.trailing_stop_bonus > 0.0, f"stage {idx}: letting a winner run is unrewarded"
    assert r.agent_close_bonus < 0.0, f"stage {idx}: closing by hand is not discouraged"


def test_the_bonus_decays_to_zero_past_the_window():
    for idx, s in enumerate(STAGES):
        cfg = s.rewards
        assert _bonus(cfg, cfg.max_trade_bars_for_bonus) == pytest.approx(0.0, abs=1e-12)
        assert _bonus(cfg, cfg.max_trade_bars_for_bonus * 2) == pytest.approx(0.0, abs=1e-12), (
            f"stage {idx}: holding forever still earns a bonus"
        )


def test_the_incentive_survives_into_the_env_config():
    """The ladder is worthless if _apply_overrides_to_object drops it."""
    from envs.core.env_types import PropFirmConfig

    cfg = PropFirmConfig()
    for field in (
        "time_efficiency_enabled",
        "time_efficiency_scale",
        "optimal_trade_bars",
        "max_trade_bars_for_bonus",
        "trailing_stop_bonus",
        "agent_close_bonus",
    ):
        assert hasattr(cfg.reward, field), (
            f"{field} has no home on RewardConfig, so the stage value is "
            f"silently discarded"
        )
