"""Focused regressions for executable curriculum-manager semantics."""

from __future__ import annotations

import copy

import pytest

from envs.curriculum.config import CurriculumStage
from envs.curriculum.curriculum_config import get_stage_config
from envs.curriculum.curriculum_manager import CurriculumManager
from envs.curriculum.metrics import EpisodeMetrics
from envs.curriculum.protocols import RecoveryProtocolState
from envs.curriculum.validation_gates import ValidationGateResult


def _manager(stage: CurriculumStage) -> CurriculumManager:
    return CurriculumManager(
        initial_stage=stage,
        auto_promote=False,
        auto_demote=False,
        verbose=False,
        rng_seed=7,
    )


def _cache_effective_stage(
    manager: CurriculumManager,
    stage: CurriculumStage,
) -> None:
    manager._cached_effective_stage = stage
    manager._cached_effective_stage_episode = manager.total_episodes + 1


def _selective_episode(*, passing: bool = True, win_rate: float = 0.0) -> EpisodeMetrics:
    return EpisodeMetrics(
        trade_count=1 if passing else 4,
        avg_setup_quality=0.80,
        avg_entry_certainty=0.80,
        win_rate=win_rate,
    )


def test_effective_config_uses_the_episode_cached_stage() -> None:
    manager = _manager(CurriculumStage.INTEGRATOR)
    _cache_effective_stage(manager, CurriculumStage.EXPLORER)

    effective = manager.get_effective_stage_config()

    assert manager.current_stage is CurriculumStage.INTEGRATOR
    assert manager.get_effective_stage() is CurriculumStage.EXPLORER
    assert effective.stage is CurriculumStage.EXPLORER
    assert effective.rewards.reward_scale == get_stage_config(
        CurriculumStage.EXPLORER
    ).rewards.reward_scale


def test_blend_recovery_and_constraints_are_materialized_into_overrides() -> None:
    manager = _manager(CurriculumStage.EXPERIMENTER)
    current = get_stage_config(CurriculumStage.EXPERIMENTER)
    previous = get_stage_config(CurriculumStage.EXPLORER)
    manager._previous_stage_config = previous
    manager._previous_stage_name = CurriculumStage.EXPLORER.name
    manager._reward_blend_remaining = current.transition.reward_blend_episodes // 2
    manager._recovery_state = RecoveryProtocolState(
        triggered=True,
        episodes_remaining=3,
        reward_modifications={
            "churn_penalty_per_trade": 1.5,
            "loss_streak_caution_base": 2.0,
        },
        constraint_modifications={"max_trades_per_day": 0.5},
    )
    _cache_effective_stage(manager, CurriculumStage.EXPERIMENTER)

    effective = manager.get_effective_stage_config()
    alpha = manager.reward_blend_factor
    blended_churn = (
        previous.rewards.churn_penalty_per_trade * (1.0 - alpha)
        + current.rewards.churn_penalty_per_trade * alpha
    )
    blended_caution = (
        previous.rewards.loss_streak_caution_base * (1.0 - alpha)
        + current.rewards.loss_streak_caution_base * alpha
    )

    assert effective.reward_overrides is not None
    assert effective.env_overrides is not None
    assert effective.rewards.churn_penalty_per_trade == pytest.approx(blended_churn * 1.5)
    assert effective.reward_overrides["churn_penalty_per_trade"] == pytest.approx(
        effective.rewards.churn_penalty_per_trade
    )
    assert effective.reward_overrides["loss_streak_caution_base"] == pytest.approx(
        blended_caution * 2.0
    )
    assert effective.constraints.max_trades_per_day == 15
    assert effective.env_overrides["max_trades_per_day"] == 15


def test_selectivity_success_rate_is_a_phase_ratio_not_episode_win_rate() -> None:
    manager = _manager(CurriculumStage.STRATEGIST)
    manager.start_selectivity_phase(
        episodes=5,
        requirements={"required_success_rate": 0.60},
    )

    outcomes = [True, False, True, False, True]
    final = {}
    for passing in outcomes:
        # A zero win rate is intentional: required_success_rate describes how
        # many episodes meet the selectivity criteria, not trading win rate.
        final = manager.check_selectivity_phase(
            _selective_episode(passing=passing, win_rate=0.0)
        )

    assert final["phase_complete"] is True
    assert final["phase_passed"] is True
    assert final["successful_episodes"] == 3
    assert final["failed_episodes"] == 2
    assert final["success_rate"] == pytest.approx(0.60)
    assert manager._selectivity_phase_active is False
    assert manager._selectivity_phase_completed_epoch == manager.current_stage_epoch


def test_selectivity_waits_for_full_evidence_and_zero_trade_cannot_fake_quality() -> None:
    manager = _manager(CurriculumStage.STRATEGIST)
    manager.start_selectivity_phase(
        episodes=3,
        requirements={"required_success_rate": 1.0},
    )

    first = manager.check_selectivity_phase(_selective_episode())
    second = manager.check_selectivity_phase(_selective_episode())
    assert "phase_complete" not in first
    assert "phase_complete" not in second

    no_trade = EpisodeMetrics(
        trade_count=0,
        avg_setup_quality=0.99,
        avg_entry_certainty=0.99,
        win_rate=1.0,
    )
    final = manager.check_selectivity_phase(no_trade)

    assert final["evidence_complete"] is True
    assert final["phase_passed"] is False
    assert final["phase_reset"] is True
    assert final["successful_episodes"] == 2
    assert any("No executed trade" in item for item in final["violations"])
    assert manager._selectivity_phase_active is True
    assert manager._selectivity_phase_episodes_remaining == 3


def test_selectivity_state_round_trip_and_legacy_target_are_fail_closed() -> None:
    manager = _manager(CurriculumStage.STRATEGIST)
    manager.start_selectivity_phase(
        episodes=5,
        requirements={"required_success_rate": 0.60},
    )
    manager.check_selectivity_phase(_selective_episode())
    manager.check_selectivity_phase(_selective_episode(passing=False))

    state = manager.to_dict()
    restored = CurriculumManager.from_dict(
        copy.deepcopy(state),
        auto_promote=False,
        auto_demote=False,
        verbose=False,
    )
    assert restored._selectivity_phase_episodes_remaining == 3
    assert restored._selectivity_phase_target_episodes == 5
    assert restored._selectivity_phase_failures == 1

    legacy_state = copy.deepcopy(state)
    legacy_state["selectivity_phase"].pop("target_episodes")
    legacy = CurriculumManager.from_dict(
        legacy_state,
        auto_promote=False,
        auto_demote=False,
        verbose=False,
    )
    assert legacy._selectivity_phase_target_episodes >= (
        legacy._selectivity_phase_episodes_remaining
        + legacy._selectivity_phase_failures
    )


def test_failed_validation_retries_are_spaced_and_spacing_survives_state_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict] = []

    def evaluator(_scenarios, training_stats):
        calls.append(dict(training_stats))
        return {}

    manager = CurriculumManager(
        initial_stage=CurriculumStage.INTEGRATOR,
        auto_promote=True,
        auto_demote=False,
        verbose=False,
        validation_gate_evaluator=evaluator,
    )
    manager._transition_cooldown_remaining = 0
    monkeypatch.setattr(manager, "check_promotion_criteria", lambda: (True, {}))
    monkeypatch.setattr(
        manager._validation_gate,
        "evaluate_all",
        lambda **_kwargs: ValidationGateResult(
            scenarios_passed=0,
            scenarios_total=1,
            pass_rate=0.0,
            performance_ratio=0.0,
            gate_passed=False,
            blocking_reasons=["synthetic failure"],
        ),
    )

    assert manager.try_promote() == (False, None)
    assert len(calls) == 1
    assert calls[0]["mean_trade_count"] == pytest.approx(0.0)

    saved = manager.to_dict()
    restored = CurriculumManager.from_dict(
        saved,
        auto_promote=True,
        auto_demote=False,
        verbose=False,
        validation_gate_evaluator=evaluator,
    )
    restored._transition_cooldown_remaining = 0
    monkeypatch.setattr(restored, "check_promotion_criteria", lambda: (True, {}))
    monkeypatch.setattr(
        restored._validation_gate,
        "evaluate_all",
        manager._validation_gate.evaluate_all,
    )

    interval = restored._validation_retry_interval(restored.stage_config)
    restored.stage_episodes = interval - 1
    assert restored.try_promote() == (False, None)
    assert len(calls) == 1

    restored.stage_episodes = interval
    assert restored.try_promote() == (False, None)
    assert len(calls) == 2


def test_terminal_mastery_requires_full_budget_competence_and_external_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation_calls = 0

    def evaluator(_scenarios, _training_stats):
        nonlocal validation_calls
        validation_calls += 1
        return {}

    manager = CurriculumManager(
        initial_stage=CurriculumStage.LIVE_READY,
        auto_promote=True,
        auto_demote=False,
        verbose=False,
        validation_gate_evaluator=evaluator,
    )
    manager._transition_cooldown_remaining = 0
    monkeypatch.setattr(manager, "check_promotion_criteria", lambda: (True, {}))
    monkeypatch.setattr(manager, "check_demotion_criteria", lambda: (False, {}))
    monkeypatch.setattr(
        manager._validation_gate,
        "evaluate_all",
        lambda **_kwargs: ValidationGateResult(
            scenarios_passed=8,
            scenarios_total=8,
            pass_rate=1.0,
            performance_ratio=1.0,
            gate_passed=True,
        ),
    )

    manager.stage_episodes = 100
    manager.stage_timesteps = manager.stage_config.competence.min_timesteps
    stopped, _reason = manager.should_stop_training(mastery_confirmation_episodes=100)
    assert stopped is False
    assert validation_calls == 0

    manager.stage_episodes = manager.stage_config.competence.min_episodes
    manager.stage_timesteps = manager.stage_config.competence.min_timesteps
    stopped, reason = manager.should_stop_training(mastery_confirmation_episodes=100)
    assert stopped is True
    assert "met competence and validation gates" in reason
    assert validation_calls == 1

    no_evaluator = _manager(CurriculumStage.LIVE_READY)
    no_evaluator._transition_cooldown_remaining = 0
    no_evaluator.stage_episodes = no_evaluator.stage_config.competence.min_episodes
    no_evaluator.stage_timesteps = no_evaluator.stage_config.competence.min_timesteps
    monkeypatch.setattr(no_evaluator, "check_promotion_criteria", lambda: (True, {}))
    monkeypatch.setattr(no_evaluator, "check_demotion_criteria", lambda: (False, {}))

    stopped, _reason = no_evaluator.should_stop_training(mastery_confirmation_episodes=100)
    assert stopped is False
