"""Abstention is neutral, while selective promotion still requires evidence."""

from __future__ import annotations

from typing import Iterable

import pytest

from envs.core.env_types import PropFirmConfig
from envs.curriculum.curriculum_config import CurriculumStage
from envs.curriculum.curriculum_manager import CurriculumManager
from envs.curriculum.metrics import EpisodeMetrics
from envs.curriculum.validation_gates import (
    ValidationGateChecker,
    ValidationScenario,
    ValidationScenarioType,
)


def test_every_curriculum_stage_keeps_the_legacy_inactivity_penalty_disabled():
    from envs.curriculum import CurriculumStage
    from envs.curriculum.curriculum_config import get_stage_config

    for stage in CurriculumStage:
        assert get_stage_config(stage).rewards.min_trades_penalty == 0.0
from envs.prop_firm_env import PropFirmTradingEnv


def _episode(*, pnl: float = 0.0, trades: int = 0, breach: bool = False) -> EpisodeMetrics:
    active = trades > 0
    return EpisodeMetrics(
        total_pnl=pnl,
        win_rate=(2.0 / 3.0) if active else 0.0,
        trade_count=trades,
        winning_trades=2 if active else 0,
        losing_trades=1 if active else 0,
        max_drawdown=0.02 if breach else 0.0,
        daily_drawdown=0.02 if breach else 0.0,
        dd_breach=breach,
        avg_r_multiple=0.20 if active else 0.0,
        profit_factor=2.0 if active else 0.0,
        avg_entry_quality=0.60 if active else 0.50,
        episode_length=1500,
    )


def _experimenter_with_window(window: Iterable[EpisodeMetrics]) -> CurriculumManager:
    episodes = list(window)
    manager = CurriculumManager(
        initial_stage=CurriculumStage.EXPERIMENTER,
        auto_promote=False,
        auto_demote=False,
        verbose=False,
        rng_seed=7,
    )
    required = manager.stage_config.competence.min_episodes
    for _ in range(required - len(episodes)):
        manager.record_episode(_episode(), timesteps=1500)
    for episode in episodes:
        manager.record_episode(episode, timesteps=1500)
    return manager


def _trade_activity(manager: CurriculumManager) -> tuple[bool, dict]:
    promotion_ready, result = manager.check_promotion_criteria()
    return promotion_ready, result["checks"]["trade_activity"]


def test_flat_and_low_activity_have_the_same_neutral_reward(market_data):
    def run(injected_trades: int) -> tuple[float, dict]:
        cfg = PropFirmConfig()
        cfg.max_steps_per_episode = 300
        cfg.mirror_augmentation_prob = 0.0
        cfg.high_vol_oversample_prob = 0.0
        cfg.domain_randomization_enabled = False
        cfg.reward.activity_consistency_enabled = True
        cfg.reward.target_trades_per_1k_steps = 10.0
        cfg.reward.per_step_shaping_enabled = False
        env = PropFirmTradingEnv(market_data, cfg)
        try:
            env.reset(seed=91)
            total_reward = 0.0
            for step in range(cfg.max_steps_per_episode):
                if step == cfg.max_steps_per_episode - 1:
                    env.total_trades = injected_trades
                _, reward, terminated, truncated, _ = env.step(env._ACTION_HOLD)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            return total_reward, dict(env._episode_reward_components)
        finally:
            env.close()

    flat_reward, flat_components = run(0)
    low_reward, low_components = run(1)  # 1 / 3 expected trades: deliberately selective.

    assert low_reward == pytest.approx(flat_reward)
    assert "activity_consistency_penalty" not in flat_components | low_components
    assert "overactivity_penalty" not in flat_components | low_components


def test_clustered_microprofit_cannot_waive_trade_evidence():
    window = [_episode() for _ in range(40)] + [
        _episode(pnl=0.01, trades=3) for _ in range(10)
    ]
    ready, activity = _trade_activity(_experimenter_with_window(window))

    assert not ready
    assert not activity["passed"]
    assert not activity["selective_evidence"]
    assert activity["eligible_episodes"] == 10
    assert activity["eligible_trades"] == 30


def test_distributed_positive_edge_can_waive_the_mean_trade_floor():
    window = [_episode() for _ in range(25)] + [
        _episode(pnl=20.0 + i % 3, trades=3) for i in range(25)
    ]
    ready, activity = _trade_activity(_experimenter_with_window(window))

    assert ready
    assert activity["actual_mean"] < activity["required_mean"]
    assert activity["selective_evidence"]
    assert activity["eligible_episodes"] >= activity["required_eligible_episodes"]
    assert activity["eligible_trades"] >= activity["required_eligible_trades"]
    assert activity["pnl_mean_ci_low"] > 0.0
    assert activity["dd_breach_rate"] == 0.0


@pytest.mark.parametrize("failure", ["ci", "breach"])
def test_selective_waiver_fails_without_positive_ci_or_with_any_breach(failure):
    active = []
    for i in range(25):
        pnl = (100.0 if i % 2 == 0 else -100.0) if failure == "ci" else 20.0
        active.append(_episode(pnl=pnl, trades=3, breach=(failure == "breach" and i == 0)))
    manager = _experimenter_with_window([_episode() for _ in range(25)] + active)
    ready, activity = _trade_activity(manager)

    assert not ready
    assert not activity["selective_evidence"]
    if failure == "ci":
        assert activity["pnl_mean_ci_low"] <= 0.0
    else:
        assert activity["dd_breach_rate"] > 0.0


def _validation_episodes(pnls: Iterable[float]) -> list[dict]:
    return [
        {
            "total_pnl": float(pnl),
            "trade_count": 5,
            "win_rate": 0.60,
            "profit_factor": 1.5,
            "avg_r_multiple": 0.20,
            "max_drawdown": 0.01,
            "dd_breach": False,
        }
        for pnl in pnls
    ]


def _standard_scenario() -> ValidationScenario:
    return ValidationScenario(
        name="standard_validation",
        scenario_type=ValidationScenarioType.STANDARD,
        description="paired held-out paths",
        min_episodes=10,
        min_trades=50,
    )


def test_standard_validation_accepts_absolute_evidence_despite_lower_activity():
    result = ValidationGateChecker().evaluate_scenario(
        _standard_scenario(),
        _validation_episodes(range(100, 110)),
        {"mean_trade_count": 100.0},
    )

    assert result.total_trades == 50
    assert result.pnl_mean_ci_low > 0.0
    assert result.passed
    assert not any("Under-trading" in reason for reason in result.failure_reasons)


def test_standard_validation_rejects_a_pnl_interval_crossing_zero():
    result = ValidationGateChecker().evaluate_scenario(
        _standard_scenario(),
        _validation_episodes([100.0, -100.0] * 5),
        {"mean_trade_count": 5.0},
    )

    assert result.total_trades == 50
    assert result.pnl_mean_ci_low <= 0.0
    assert not result.passed
    assert any("No positive edge evidence" in reason for reason in result.failure_reasons)


def test_final_gate_cannot_hide_a_failed_execution_stress_scenario() -> None:
    checker = ValidationGateChecker()
    scenarios = checker.get_scenarios_for_stage(8)
    validation_results = {}
    for scenario in scenarios:
        count = scenario.min_episodes
        trades = 1 if scenario.name in {"low_volatility_patience", "fomo_resistance"} else 10
        pnl = -10.0 if scenario.name == "stress_test" else 100.0
        validation_results[scenario.name] = [
            {
                "total_pnl": pnl,
                "trade_count": trades,
                "win_rate": 0.70,
                "profit_factor": 1.60,
                "avg_r_multiple": 0.30,
                "max_drawdown": 0.01,
                "avg_bars_between_trades": 12.0,
                "avg_setup_quality": 0.75,
                "avg_entry_certainty": 0.80,
                "fomo_trade_count": 0,
                "revenge_trade_count": 0,
                "dd_breach": False,
                "episode_start_index": episode_index * 300,
                "episode_end_index": episode_index * 300 + 168,
                "episode_length": 168,
            }
            for episode_index in range(count)
        ]

    result = checker.evaluate_all(
        validation_results=validation_results,
        training_stats={
            "mean_win_rate": 0.65,
            "mean_profit_factor": 1.50,
            "mean_r_multiple": 0.25,
            "mean_trade_count": 10.0,
            "total_trades": 500,
        },
        stage_name="PROFESSIONAL",
        stage_index=8,
        scenarios=scenarios,
    )

    assert result.scenarios_passed == len(scenarios) - 1
    assert result.gate_passed is False
    assert any(
        "Final-stage scenario failures" in reason and "stress_test" in reason
        for reason in result.blocking_reasons
    )


def test_promotion_gate_rejects_overlapping_episode_evidence() -> None:
    checker = ValidationGateChecker()
    scenario = _standard_scenario()
    episodes = _validation_episodes([100.0] * scenario.min_episodes)
    for index, episode in enumerate(episodes):
        # Each interval overlaps the next even though every record has a
        # syntactically complete identity.
        episode["episode_start_index"] = index * 32
        episode["episode_end_index"] = index * 32 + 168
        episode["episode_length"] = 168

    result = checker.evaluate_all(
        validation_results={scenario.name: episodes},
        training_stats={
            "mean_win_rate": 0.65,
            "mean_profit_factor": 1.50,
            "mean_r_multiple": 0.25,
            "mean_trade_count": 5.0,
            "total_trades": 100,
        },
        stage_name="INTEGRATOR",
        stage_index=5,
        scenarios=[scenario],
    )

    assert result.gate_passed is False
    assert result.scenarios_passed == 0
    assert result.scenario_results[scenario.name].overlapping_episodes > 0
    assert any(
        "Overlapping episode evidence" in reason
        for reason in result.blocking_reasons
    )
