"""Observation-contract tests.

Regression cover for the defect that made the agent blind:

  PPO_OBS_SIZE was 84 while _build_feature_names() produced 106 names, so the
  module raised at import. envs/prop_firm_env.py caught that with a bare
  `except Exception`, set PPO_OBS_SIZE = 90, and fell back to np.zeros(90).
  Training ran normally and every dashboard metric looked healthy.

Each test below fails if any part of that chain is reintroduced.
"""
from __future__ import annotations

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────
# Schema self-consistency
# ─────────────────────────────────────────────────────────────

def test_schema_size_matches_feature_names_and_groups():
    """The three independent statements of observation width must agree."""
    from modules.meta.ppo_observation_builder import (
        FEATURE_GROUPS,
        PPO_OBS_FEATURE_NAMES,
        PPO_OBS_SIZE,
    )

    max_group_end = max(end for _start, end in FEATURE_GROUPS.values())
    assert len(PPO_OBS_FEATURE_NAMES) == PPO_OBS_SIZE, (
        f"feature-name count {len(PPO_OBS_FEATURE_NAMES)} != PPO_OBS_SIZE {PPO_OBS_SIZE}"
    )
    assert max_group_end == PPO_OBS_SIZE, (
        f"FEATURE_GROUPS span {max_group_end} != PPO_OBS_SIZE {PPO_OBS_SIZE}"
    )


def test_feature_groups_are_contiguous_and_non_overlapping():
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS, PPO_OBS_SIZE

    spans = sorted(FEATURE_GROUPS.values())
    assert spans[0][0] == 0, "feature groups must start at index 0"
    for (_, prev_end), (next_start, _) in zip(spans, spans[1:]):
        assert prev_end == next_start, f"gap or overlap at index {prev_end}"
    assert spans[-1][1] == PPO_OBS_SIZE


def test_feature_names_are_unique():
    from modules.meta.ppo_observation_builder import PPO_OBS_FEATURE_NAMES

    dupes = {n for n in PPO_OBS_FEATURE_NAMES if PPO_OBS_FEATURE_NAMES.count(n) > 1}
    assert not dupes, f"duplicate feature names: {sorted(dupes)}"


# ─────────────────────────────────────────────────────────────
# The builder must be mandatory, not optional
# ─────────────────────────────────────────────────────────────

def test_observation_builder_imports_without_torch():
    """The builder needs only numpy. Coupling it to torch is what forced the
    try/except that hid the schema break."""
    import importlib

    mod = importlib.import_module("modules.meta.ppo_observation_builder")
    assert mod.PPO_OBS_SIZE > 0


def test_env_has_no_silent_observation_fallback(env):
    """A missing observation must raise, never degrade to zeros."""
    assert env.obs_builder is not None, "env must always hold a real builder"
    with pytest.raises(RuntimeError, match="fallback"):
        env._fallback_observation()


def test_env_observation_space_matches_builder_width(env):
    from modules.meta.ppo_observation_builder import PPO_OBS_SIZE

    assert env.observation_space.shape == (PPO_OBS_SIZE,)


# ─────────────────────────────────────────────────────────────
# Runtime health — the check that would have caught the outage
# ─────────────────────────────────────────────────────────────

def test_observation_is_not_constant(rollout):
    """THE canary. A blind agent produces a constant vector."""
    assert rollout.shape[0] > 100, "rollout too short to judge"
    assert float(rollout.std()) > 1e-6, (
        "observation is constant across the entire rollout - the agent is blind"
    )


def test_observation_has_no_nan_or_inf(rollout):
    assert not np.isnan(rollout).any(), "NaN in observation"
    assert not np.isinf(rollout).any(), "inf in observation"


def test_at_least_half_the_dimensions_carry_information(rollout):
    """Dead dims are tolerated (documented stubs) but must not dominate."""
    std = rollout.std(axis=0)
    live = int((std > 1e-9).sum())
    total = rollout.shape[1]
    assert live >= total // 2, (
        f"only {live}/{total} observation dims vary; the input is mostly dead"
    )


def test_price_block_is_always_live(rollout):
    """Market features are never legitimately constant."""
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    start, end = FEATURE_GROUPS["m15_price"]
    std = rollout[:, start:end].std(axis=0)
    dead = int((std < 1e-9).sum())
    assert dead == 0, f"{dead} of {end - start} m15_price dims are constant"


def test_observation_values_are_bounded(rollout):
    """Every documented feature is clipped; unbounded values signal a scaling bug."""
    finite_max = float(np.abs(rollout).max())
    assert finite_max < 1e4, f"observation magnitude {finite_max:.3g} is implausible"
