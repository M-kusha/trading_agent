from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from envs.curriculum import CurriculumStage, get_stage_config
from train.train_prop_firm import (
    assess_terminal_mastery,
    build_curriculum_budget_preflight,
    build_curriculum_run_provenance,
    enforce_goal_based_budget,
)


def _manifest(role: str, fingerprint: str) -> dict:
    return {
        "role": role,
        "dataset_fingerprint": fingerprint,
        "boundary": "2026-06-15T00:00:00+00:00",
        "git_head": "abc123",
        "training_code_sha256": "code456",
        "frames": [
            {
                "instrument": "XAUUSD",
                "timeframe": "M15",
                "rows": 10,
                "first_time": "2026-06-01T00:00:00+00:00",
                "last_time": "2026-06-15T00:00:00+00:00",
                "content_sha256": fingerprint,
            }
        ],
    }


def test_budget_uses_stage_local_thresholds_and_declared_current_stage_share() -> None:
    plan = build_curriculum_budget_preflight(
        start_stage="EXPLORER",
        safety_cap_timesteps=100_000_000,
    )
    stages = {entry["stage"]: entry for entry in plan["stages"]}

    explorer = stages["EXPLORER"]
    assert explorer["local_min_timesteps"] == 100_000
    assert explorer["declared_current_stage_probability"] == 1.0
    assert explorer["global_timesteps_required"] == 100_000

    trend = stages["TREND_STUDENT"]
    assert trend["local_min_timesteps"] == 300_000
    assert trend["declared_current_stage_probability"] == pytest.approx(0.80)
    assert trend["global_timesteps_required"] == math.ceil(300_000 / 0.80)

    live_ready = stages["LIVE_READY"]
    live_config = get_stage_config(CurriculumStage.LIVE_READY)
    live_weights = live_config.mixed_stage_sampling
    live_probability = live_weights.current_stage_weight / (
        live_weights.current_stage_weight
        + live_weights.recent_stages_weight
        + live_weights.foundation_weight
    )
    assert live_ready["local_min_timesteps"] == live_config.competence.min_timesteps
    assert live_ready["global_timesteps_required"] == math.ceil(
        live_config.competence.min_timesteps / live_probability
    )
    assert plan["terminal_mastery_possible"] is True


def test_goal_based_cap_one_step_below_floor_is_rejected() -> None:
    complete_plan = build_curriculum_budget_preflight(
        start_stage="EXPLORER",
        safety_cap_timesteps=100_000_000,
    )
    floor = complete_plan["minimum_total_target_timesteps"]
    underfunded = build_curriculum_budget_preflight(
        start_stage="EXPLORER",
        safety_cap_timesteps=floor - 1,
    )

    assert underfunded["terminal_mastery_possible"] is False
    assert underfunded["shortfall_timesteps"] == 1
    with pytest.raises(ValueError, match=r"--goal-based safety cap cannot reach LIVE_READY"):
        enforce_goal_based_budget(underfunded, goal_based=True)

    # The same fixed budget is allowed, but its theoretical ceiling remains
    # explicit instead of being presented as terminal mastery.
    enforce_goal_based_budget(underfunded, goal_based=False)
    assert underfunded["maximum_theoretical_stage"] == "LIVE_READY"
    assert underfunded["first_unfunded_stage"] == "LIVE_READY"


def test_fixed_run_reports_the_highest_stage_the_budget_can_enter() -> None:
    explorer_only = build_curriculum_budget_preflight(
        start_stage="EXPLORER",
        safety_cap_timesteps=99_999,
    )
    assert explorer_only["maximum_theoretical_stage"] == "EXPLORER"
    assert explorer_only["first_unfunded_stage"] == "EXPLORER"

    reaches_experimenter = build_curriculum_budget_preflight(
        start_stage="EXPLORER",
        safety_cap_timesteps=100_000,
    )
    assert reaches_experimenter["maximum_theoretical_stage"] == "EXPERIMENTER"
    assert reaches_experimenter["first_unfunded_stage"] == "EXPERIMENTER"


def test_resume_budget_credits_only_current_stage_progress() -> None:
    plan = build_curriculum_budget_preflight(
        start_stage="INTEGRATOR",
        safety_cap_timesteps=2_000_000,
        already_consumed_timesteps=1_000_000,
        current_stage_timesteps=200_000,
    )
    first = plan["stages"][0]
    assert first["stage"] == "INTEGRATOR"
    assert first["local_timesteps_already_recorded"] == 200_000
    assert first["local_timesteps_remaining"] == 500_000
    assert plan["available_timesteps"] == 1_000_000


def test_nonterminal_provenance_is_fail_closed_and_reproducible() -> None:
    provenance = build_curriculum_run_provenance(
        loaded_manifest=_manifest("loaded", "loaded-hash"),
        train_manifest=_manifest("curriculum_train", "train-hash"),
        holdout_manifest=_manifest("curriculum_holdout_with_context", "holdout-hash"),
        final_stage="INTEGRATOR",
        run_outcome="finished",
        actual_timesteps=3_000_000,
        start_stage="EXPLORER",
        goal_based=False,
        safety_cap_timesteps=3_000_000,
        holdout_enabled=True,
        holdout_ratio=0.15,
        holdout_split_at="2026-06-15",
        resolved_holdout_split_at="2026-06-15T00:00:00+00:00",
        data_cutoff="2026-08-03",
        regime_start_at="2026-02-01",
        regime_target_share=0.40,
        observation_schema={"version": "v8", "base_width": 46},
        command=["python", "train/train_prop_firm.py", "--curriculum"],
        budget_preflight={"maximum_theoretical_stage": "RISK_MANAGER"},
        model_record={"path": "model.zip", "sha256": "model-hash", "bytes": 123},
    )

    assert provenance["status"] == "CURRICULUM_NOT_TERMINAL"
    assert provenance["accepted"] is False
    assert provenance["curriculum_complete"] is False
    assert provenance["promotion_eligible"] is False
    assert provenance["datasets"]["loaded"]["fingerprint"] == "loaded-hash"
    assert provenance["datasets"]["train"]["fingerprint"] == "train-hash"
    assert provenance["datasets"]["holdout"]["fingerprint"] == "holdout-hash"
    assert provenance["arguments"]["holdout_split_at"] == "2026-06-15"
    assert provenance["arguments"]["resolved_holdout_split_at"] == "2026-06-15T00:00:00+00:00"
    assert provenance["arguments"]["data_cutoff"] == "2026-08-03"
    assert provenance["arguments"]["regime_start_at"] == "2026-02-01"
    assert provenance["arguments"]["regime_target_share"] == 0.40
    assert provenance["observation_schema"]["version"] == "v8"
    assert provenance["command"][-1] == "--curriculum"


def test_live_ready_enum_without_mastery_evidence_remains_incomplete() -> None:
    provenance = build_curriculum_run_provenance(
        loaded_manifest=_manifest("loaded", "loaded-hash"),
        train_manifest=_manifest("train", "train-hash"),
        holdout_manifest=_manifest("holdout", "holdout-hash"),
        final_stage="LIVE_READY",
        run_outcome="finished",
        actual_timesteps=20_000_000,
        start_stage="EXPLORER",
        goal_based=True,
        safety_cap_timesteps=100_000_000,
        holdout_enabled=True,
        holdout_ratio=0.0,
        holdout_split_at="2026-06-15",
        resolved_holdout_split_at="2026-06-15T00:00:00+00:00",
        data_cutoff=None,
        regime_start_at="2026-02-01",
        regime_target_share=0.40,
        observation_schema={"version": "v8", "base_width": 46},
        command=["python"],
        budget_preflight={"terminal_mastery_possible": True},
        model_record={"path": "model.zip", "sha256": "model-hash", "bytes": 123},
    )

    assert provenance["status"] == "LIVE_READY_REACHED_MASTERY_UNCONFIRMED"
    assert provenance["terminal_stage_reached"] is True
    assert provenance["curriculum_complete"] is False
    assert provenance["accepted"] is False
    assert provenance["promotion_eligible"] is False


def test_terminal_mastery_requires_floors_transition_validation_and_stress() -> None:
    config = get_stage_config(CurriculumStage.LIVE_READY)
    transition = {
        "type": "promotion",
        "from_stage": "PROFESSIONAL",
        "to_stage": "LIVE_READY",
    }
    validation = {
        "stage": "LIVE_READY",
        "passed": True,
        "scenarios_passed": 5,
        "scenarios_total": 5,
    }
    report = {
        "recent_transitions": [transition],
        "validation_gate_history": [validation],
        "stress_test_history": [],
    }
    manager = SimpleNamespace(
        stage_config=config,
        current_stage=CurriculumStage.LIVE_READY,
        stage_episodes=config.competence.min_episodes,
        stage_timesteps=config.competence.min_timesteps,
        get_progress_report=lambda: report,
    )

    mastered = assess_terminal_mastery(
        manager,
        mastery_confirmation_episodes=100,
    )
    assert mastered["confirmed"] is True

    validation["scenarios_passed"] = 4
    missing_stress = assess_terminal_mastery(
        manager,
        mastery_confirmation_episodes=100,
    )
    assert missing_stress["confirmed"] is False
    assert missing_stress["checks"]["development_stress_gate"]["passed"] is False

    report["validation_gate_history"] = [
        {
            "stage": "PROFESSIONAL",
            "passed": True,
            "scenarios_passed": 5,
            "scenarios_total": 5,
        }
    ]
    stale_preterminal_only = assess_terminal_mastery(
        manager,
        mastery_confirmation_episodes=100,
    )
    assert stale_preterminal_only["confirmed"] is False
