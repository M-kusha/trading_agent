"""Every curriculum override must actually reach the env config.

_apply_overrides_to_object assigns an override only `if hasattr(target, k)`.
A key with no matching field used to be dropped, so a stage could look
carefully tuned while the env quietly used its own default. Runtime application
now fails before the episode instead of continuing with a partial stage.

That is not hypothetical. activity_deviation_penalty_cap was set on all ten
stages (10.0 -> 20.0 -> 10.0) and defined nowhere on the env's RewardConfig, so
`getattr(reward_cfg, "activity_deviation_penalty_cap", 2.0)` returned 2.0 for
the entire curriculum. The over-trading penalty saturated at -2.0 per episode
no matter how far the agent exceeded the stage's trade target - observed at
exactly -2.0000 across 8 consecutive episodes of a live run.

These tests compare what the stages emit against what the config can receive.
"""

from __future__ import annotations

import pytest

from envs.core.env_types import PropFirmConfig
from envs.curriculum.curriculum_config import get_all_stage_configs
from envs.prop_firm_env import PropFirmTradingEnv

# commission_per_lot has no matching attribute by design: it is translated into
# config.execution.commission_spec instead of being assigned directly.
HANDLED_ELSEWHERE = PropFirmTradingEnv._OVERRIDE_KEYS_HANDLED_ELSEWHERE

CHANNELS = ("env_overrides", "reward_overrides", "execution_overrides")


def _target_for(cfg: PropFirmConfig, channel: str):
    return {
        "env_overrides": cfg,
        "reward_overrides": cfg.reward,
        "execution_overrides": cfg.execution,
    }[channel]


@pytest.mark.parametrize("channel", CHANNELS)
def test_no_stage_emits_an_override_the_env_cannot_receive(channel):
    cfg = PropFirmConfig()
    target = _target_for(cfg, channel)

    dropped = {}
    for stage_name, stage in get_all_stage_configs().items():
        overrides = getattr(stage, channel, None)
        if not isinstance(overrides, dict):
            continue
        for key in overrides:
            if "." in str(key) or key in HANDLED_ELSEWHERE:
                continue
            if not hasattr(target, key):
                dropped.setdefault(key, []).append(str(stage_name))

    assert not dropped, (
        f"{channel} keys with no field on {type(target).__name__} - these are "
        f"silently discarded and the configured value never takes effect:\n"
        + "\n".join(f"  {k} (stages: {', '.join(v)})" for k, v in sorted(dropped.items()))
    )


def test_the_activity_cap_reaches_the_env():
    """The specific field whose loss flattened the over-trading penalty."""
    cfg = PropFirmConfig()
    assert hasattr(cfg.reward, "activity_deviation_penalty_cap")

    for stage_name, stage in get_all_stage_configs().items():
        overrides = stage.reward_overrides or {}
        cap = overrides.get("activity_deviation_penalty_cap")
        assert cap is not None, f"stage {stage_name} stopped emitting the cap"
        assert cap > 2.0, (
            f"stage {stage_name} cap is {cap}, at or below the old fallback of "
            f"2.0 - the regression would be invisible again"
        )


def test_an_unknown_override_fails_before_it_can_be_swallowed():
    """A malformed stage must stop training, not merely leave a log behind."""
    env = PropFirmTradingEnv.__new__(PropFirmTradingEnv)
    env._reported_dropped_overrides = set()
    cfg = PropFirmConfig()

    with pytest.raises(AttributeError, match="no_such_reward_field"):
        env._apply_overrides_to_object(cfg.reward, {"no_such_reward_field": 1.23})


def test_a_known_override_is_applied_and_not_reported(caplog):
    env = PropFirmTradingEnv.__new__(PropFirmTradingEnv)
    env._reported_dropped_overrides = set()
    cfg = PropFirmConfig()

    with caplog.at_level("ERROR"):
        env._apply_overrides_to_object(cfg.reward, {"activity_deviation_penalty_cap": 17.0})

    assert cfg.reward.activity_deviation_penalty_cap == 17.0
    assert "NOT in effect" not in caplog.text


def test_commission_per_lot_is_not_reported_as_dropped(caplog):
    """It is handled by an explicit branch, so it must not raise a false alarm."""
    env = PropFirmTradingEnv.__new__(PropFirmTradingEnv)
    env._reported_dropped_overrides = set()
    cfg = PropFirmConfig()

    with caplog.at_level("ERROR"):
        env._apply_overrides_to_object(cfg.execution, {"commission_per_lot": 7.0})

    assert "commission_per_lot" not in caplog.text
