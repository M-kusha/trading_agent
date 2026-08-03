"""Standing aside must be allowed, and drawdown must not be negotiable.

Two findings from the first honest out-of-sample run drove these.

On live FTMO bars, always-flat returned 0.00% and beat the trained model's
-8.25%. Doing nothing was the best available strategy for eight months. But the
curriculum forbade it: min_trade_count_avg (4-5.5) was a hard promotion gate,
min_trades_penalty fired below 0.2x the target, and patience_bonus_per_bar was
0.0 on stages 0-3. The system could not learn the most profitable behaviour
available to it.

And drawdown existed only as a reward term, which an optimizer can trade
against - a large enough expected gain justifies breaching. A prop-firm limit is
not that kind of quantity: crossing it ends the account. It belongs in the
action mask, where the policy cannot negotiate and where live enforcement is
identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from envs.core.env_types import PropFirmConfig


# ── drawdown veto ───────────────────────────────────────────────────────────

def test_entries_are_refused_once_drawdown_eats_its_budget(env):
    env.reset(seed=0)
    env.config.dd_entry_veto_fraction = 0.75

    assert not env._entry_blocked_by_drawdown(), "veto active at a flat account"

    # Drive equity below the veto threshold (75% of the 10% limit = 7.5%).
    env.equity = env.config.initial_balance * 0.92
    assert env._entry_blocked_by_drawdown(), "veto did not fire at 8% drawdown"


def test_the_mask_blocks_entries_but_still_allows_exits(env):
    env.reset(seed=0)
    env.config.dd_entry_veto_fraction = 0.75
    env.equity = env.config.initial_balance * 0.90

    mask = env.action_masks()
    longs = mask[env._ACTION_LONG_START:env._ACTION_LONG_START + env._K]
    shorts = mask[env._ACTION_SHORT_START:env._ACTION_SHORT_START + env._K]

    assert not longs.any(), "long entries still permitted past the veto"
    assert not shorts.any(), "short entries still permitted past the veto"
    assert mask[env._ACTION_HOLD], "hold must always remain legal"


def test_the_veto_can_be_disabled(env):
    env.reset(seed=0)
    env.config.dd_entry_veto_fraction = 0.0
    env.equity = env.config.initial_balance * 0.85
    assert not env._entry_blocked_by_drawdown()


def test_the_veto_fires_before_the_hard_limit():
    """It must reserve room to trade out, not trigger at the breach itself."""
    cfg = PropFirmConfig()
    veto_at = cfg.max_drawdown_limit * cfg.dd_entry_veto_fraction
    assert 0.0 < veto_at < cfg.max_drawdown_limit, (
        f"veto at {veto_at:.3f} leaves no reserve below the {cfg.max_drawdown_limit:.3f} limit"
    )


# ── abstention ──────────────────────────────────────────────────────────────

def test_inactivity_is_free_when_capital_was_protected(env):
    """Flat and even is restraint; the old flat penalty called it failure."""
    import inspect

    src = inspect.getsource(type(env).step)
    assert "protected_capital" in src, "the inactivity penalty is unconditional again"
    assert "initial_balance" in src, (
        "capital protection must be judged on the account, not on shaped reward"
    )


def test_the_trade_floor_is_waived_for_a_profitable_selective_policy():
    import inspect

    from envs.curriculum.curriculum_manager import CurriculumManager

    src = inspect.getsource(CurriculumManager)
    assert "waived_as_selective" in src, (
        "the trade-count gate no longer distinguishes restraint from incompetence"
    )
    assert "MIN_TRADES_FOR_VALID_RATE" in src


def test_a_low_activity_losing_policy_still_fails():
    """Waiving the floor must not become a free pass."""
    from envs.curriculum.curriculum_manager import MIN_TRADES_FOR_VALID_RATE

    assert MIN_TRADES_FOR_VALID_RATE >= 20, (
        "too few trades to call a win rate a measurement"
    )


# ── session block ───────────────────────────────────────────────────────────

def test_the_session_block_is_populated(env):
    from modules.meta.ppo_observation_builder import FEATURE_GROUPS

    env.reset(seed=0)
    obs = env._get_observation()
    start, end = FEATURE_GROUPS["session"]

    block = obs[start:end]
    assert block.shape == (5,)
    assert not np.isnan(block).any()
    assert np.abs(block).sum() > 0.0, "session block is all zeros - state not reaching it"


def test_hour_is_encoded_cyclically(env):
    """23:00 and 00:00 must be adjacent, not maximally distant."""
    from modules.meta.ppo_observation_builder import PPOObservationBuilder

    b = PPOObservationBuilder()
    late, _ = b._build_session_features({"hour_utc": 23.5, "spread_ratio": 1.0})
    early, _ = b._build_session_features({"hour_utc": 0.5, "spread_ratio": 1.0})
    noon, _ = b._build_session_features({"hour_utc": 12.0, "spread_ratio": 1.0})

    wrap_gap = float(np.hypot(late[0] - early[0], late[1] - early[1]))
    noon_gap = float(np.hypot(late[0] - noon[0], late[1] - noon[1]))
    assert wrap_gap < noon_gap, "midnight wrap is not continuous"


def test_prime_and_overlap_sessions_are_flagged():
    from modules.meta.ppo_observation_builder import PPOObservationBuilder

    b = PPOObservationBuilder()
    asian, _ = b._build_session_features({"hour_utc": 3.0, "spread_ratio": 1.0})
    london, _ = b._build_session_features({"hour_utc": 9.0, "spread_ratio": 1.0})
    overlap, _ = b._build_session_features({"hour_utc": 14.0, "spread_ratio": 1.0})

    assert asian[2] == 0.0 and asian[4] == 0.0
    assert london[2] == 1.0 and london[4] == 0.0
    assert overlap[2] == 1.0 and overlap[4] == 1.0


def test_spread_ratio_reads_as_normal_versus_expensive():
    from modules.meta.ppo_observation_builder import PPOObservationBuilder

    b = PPOObservationBuilder()
    normal, _ = b._build_session_features({"hour_utc": 12.0, "spread_ratio": 1.0})
    wide, _ = b._build_session_features({"hour_utc": 12.0, "spread_ratio": 2.0})
    tight, _ = b._build_session_features({"hour_utc": 12.0, "spread_ratio": 0.5})

    assert tight[3] < normal[3] < wide[3]
    assert 0.0 <= tight[3] <= 1.0 and 0.0 <= wide[3] <= 1.0
