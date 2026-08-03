from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, Optional

import numpy as np
import pytest

import train.callbacks.curriculum_callback as callback_module
from envs.curriculum.config import CurriculumStage
from train.callbacks.curriculum_callback import CurriculumTrainingCallback


class _FakeOptimizer:
    def __init__(self, learning_rate: float) -> None:
        self.param_groups = [{"lr": learning_rate}, {"lr": learning_rate}]


class _FakeModel:
    def __init__(self, env: Any = None, learning_rate: float = 8.5e-5) -> None:
        self._env = env
        self.learning_rate = learning_rate
        self.lr_schedule = lambda _: learning_rate
        self.policy = SimpleNamespace(optimizer=_FakeOptimizer(learning_rate))

    def get_env(self) -> Any:
        return self._env

    def predict(self, obs: Any, deterministic: bool, action_masks: Any):
        assert deterministic is True
        assert action_masks is not None
        return np.asarray([0]), None


class _ValidationBaseEnv:
    def __init__(self, *, supports_difficulty: bool = True) -> None:
        self.config = SimpleNamespace(max_steps_per_episode=5)
        self._scenario_data_difficulty = None
        self._data_difficulty = None
        self.seen_at_reset = []
        if not supports_difficulty:
            self.set_scenario_data_difficulty = None  # type: ignore[assignment]
            self.set_data_difficulty = None  # type: ignore[assignment]

    def set_scenario_data_difficulty(self, difficulty: Any) -> None:
        self._scenario_data_difficulty = difficulty


class _ValidationVecEnv:
    num_envs = 1

    def __init__(
        self,
        base_env: _ValidationBaseEnv,
        episode_stats: Dict[str, Any],
        top_level_info: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.envs = [base_env]
        self.base_env = base_env
        self.episode_stats = episode_stats
        self.top_level_info = top_level_info or {}

    def reset(self):
        self.base_env.seen_at_reset.append(self.base_env._scenario_data_difficulty)
        if self.base_env._scenario_data_difficulty is not None:
            self.base_env._data_difficulty = self.base_env._scenario_data_difficulty
        return np.zeros((1, 1), dtype=np.float32)

    def step(self, action: Any):
        return (
            np.zeros((1, 1), dtype=np.float32),
            np.asarray([1.0]),
            np.asarray([True]),
            [{
                **self.top_level_info,
                "episode_stats": self.episode_stats,
                "episode": {"r": 3.5},
            }],
        )


class _Manager:
    def __init__(self, lr_multiplier: float = 1.0) -> None:
        self.lr_multiplier = lr_multiplier
        self.current_stage = CurriculumStage.INTEGRATOR
        self.stage_config = SimpleNamespace(
            validation=SimpleNamespace(validation_episodes=1),
            data_difficulty=None,
        )
        self.recorded: Optional[Dict[str, Any]] = None
        self.stop_kwargs: Optional[Dict[str, Any]] = None

    def get_lr_multiplier(self) -> float:
        return self.lr_multiplier

    def step_transition_state(self, timesteps: int) -> None:
        return None

    def record_episode_from_info(self, **kwargs: Any) -> None:
        self.recorded = kwargs

    def should_stop_training(self, **kwargs: Any):
        self.stop_kwargs = kwargs
        return False, ""


def _callback(manager: Any = None, **kwargs: Any) -> CurriculumTrainingCallback:
    return CurriculumTrainingCallback(
        curriculum_manager=manager,
        total_timesteps=100_000,
        enable_lr_warmup=False,
        enable_entropy_schedule=False,
        enable_adaptive_clip_range=False,
        enable_adaptive_lr=False,
        **kwargs,
    )


def _quiet_step_methods(cb: CurriculumTrainingCallback) -> None:
    cb._apply_lr_warmup = lambda: None  # type: ignore[method-assign]
    cb._apply_entropy_schedule = lambda: None  # type: ignore[method-assign]
    cb._apply_adaptive_clip_range = lambda: None  # type: ignore[method-assign]
    cb._apply_adaptive_vf_coef = lambda: None  # type: ignore[method-assign]
    cb._apply_adaptive_learning_rate = lambda: None  # type: ignore[method-assign]
    cb._update_ppo_diagnostics = lambda: None  # type: ignore[method-assign]
    cb._maybe_save_live_metrics = lambda force=False: None  # type: ignore[method-assign]
    cb._health_watchdog = None  # type: ignore[assignment]


def test_inactive_warmup_preserves_adaptive_lr_and_active_warmup_syncs_all_views() -> None:
    manager = _Manager(lr_multiplier=1.0)
    cb = CurriculumTrainingCallback(
        curriculum_manager=manager,
        total_timesteps=100_000,
        enable_lr_warmup=True,
    )
    model = _FakeModel(learning_rate=8.5e-5)
    cb.model = model  # type: ignore[assignment]
    cb._base_lr = 1.0e-4
    cb._current_lr = 8.5e-5

    cb._apply_lr_warmup()
    assert model.learning_rate == pytest.approx(8.5e-5)
    assert model.lr_schedule(0.5) == pytest.approx(8.5e-5)
    assert {g["lr"] for g in model.policy.optimizer.param_groups} == {8.5e-5}

    manager.lr_multiplier = 0.5
    cb._apply_lr_warmup()
    assert model.learning_rate == pytest.approx(5.0e-5)
    assert model.lr_schedule(0.5) == pytest.approx(5.0e-5)
    assert {g["lr"] for g in model.policy.optimizer.param_groups} == {5.0e-5}

    manager.lr_multiplier = 1.0
    cb._apply_lr_warmup()
    assert model.learning_rate == pytest.approx(1.0e-4)
    cb._set_model_learning_rate(7.5e-5)
    cb._apply_lr_warmup()
    assert model.learning_rate == pytest.approx(7.5e-5)
    assert model.lr_schedule(0.5) == pytest.approx(7.5e-5)
    assert {g["lr"] for g in model.policy.optimizer.param_groups} == {7.5e-5}


def test_validation_uses_persistent_scenario_filter_and_emits_complete_gate_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode_stats = {
        "win_rate": 0.6,
        "profit_factor": 1.4,
        "total_pnl": 120.0,
        "avg_r_multiple": 0.3,
        "trade_count": 5,
        "max_drawdown": 0.02,
        "avg_bars_between_trades": 8.0,
        "avg_setup_quality": 0.7,
        "avg_entry_certainty": 0.75,
        "fomo_trade_count": 1,
        "revenge_trade_count": 2,
        "exit_quality_distribution": {"risk_liquidation": 3},
    }
    base_env = _ValidationBaseEnv()
    vec_env = _ValidationVecEnv(
        base_env,
        episode_stats,
        top_level_info={"termination_reason": "max_drawdown_breach"},
    )
    manager = _Manager()
    cb = _callback(manager)
    cb.model = _FakeModel(vec_env)  # type: ignore[assignment]
    monkeypatch.setattr(callback_module, "SB3_MASK_UTILS_AVAILABLE", True)
    monkeypatch.setattr(
        callback_module,
        "sb3_get_action_masks",
        lambda env: np.asarray([[True]], dtype=bool),
    )

    results = cb._run_validation_gate_episodes(
        [{"name": "high_vol", "volatility_filter": "high", "min_episodes": 1}],
        {},
    )

    assert base_env.seen_at_reset[0] is not None
    assert base_env.seen_at_reset[0].volatility_percentile_range == (0.70, 1.0)
    assert base_env._scenario_data_difficulty is None
    payload = results["high_vol"][0]
    assert payload["avg_bars_between_trades"] == 8.0
    assert payload["avg_setup_quality"] == 0.7
    assert payload["avg_entry_certainty"] == 0.75
    assert payload["fomo_trade_count"] == 1
    assert payload["revenge_trade_count"] == 2
    assert payload["risk_liquidation_exits"] == 3
    assert payload["dd_breach"] is True


def test_requested_validation_filter_and_stress_fail_closed_when_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_env = _ValidationBaseEnv(supports_difficulty=False)
    vec_env = _ValidationVecEnv(base_env, {"trade_count": 1})
    cb = _callback(_Manager())
    cb.model = _FakeModel(vec_env)  # type: ignore[assignment]
    monkeypatch.setattr(callback_module, "SB3_MASK_UTILS_AVAILABLE", True)
    monkeypatch.setattr(
        callback_module,
        "sb3_get_action_masks",
        lambda env: np.asarray([[True]], dtype=bool),
    )

    validation = cb._run_validation_gate_episodes(
        [{"name": "filtered", "volatility_filter": "high"}],
        {},
    )
    stress = cb._run_stress_test_episodes(
        [{"name": "wide", "spread_multiplier": 2.0}],
        {},
        episodes_per_scenario=1,
    )

    assert validation == {"filtered": []}
    assert stress == [[]]


def test_validation_start_planner_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base_env = _ValidationBaseEnv()
    base_env.configure_validation_episode_starts = lambda **_kwargs: object()  # type: ignore[attr-defined]
    vec_env = _ValidationVecEnv(base_env, {"trade_count": 1})
    cb = _callback(_Manager())
    cb.model = _FakeModel(vec_env)  # type: ignore[assignment]
    monkeypatch.setattr(callback_module, "SB3_MASK_UTILS_AVAILABLE", True)
    monkeypatch.setattr(
        callback_module,
        "sb3_get_action_masks",
        lambda env: np.asarray([[True]], dtype=bool),
    )

    with caplog.at_level("WARNING"):
        validation = cb._run_validation_gate_episodes(
            [{"name": "invalid_planner", "min_episodes": 1}],
            {},
        )

    assert validation == {"invalid_planner": []}
    assert "did not return a sized collection" in caplog.text


def test_validation_collection_stops_at_planned_independent_window_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_env = _ValidationBaseEnv()
    base_env.configure_validation_episode_starts = (  # type: ignore[attr-defined]
        lambda **_kwargs: [100, 200]
    )
    vec_env = _ValidationVecEnv(base_env, {"trade_count": 0})
    cb = _callback(_Manager())
    cb.model = _FakeModel(vec_env)  # type: ignore[assignment]
    monkeypatch.setattr(callback_module, "SB3_MASK_UTILS_AVAILABLE", True)
    monkeypatch.setattr(
        callback_module,
        "sb3_get_action_masks",
        lambda env: np.asarray([[True]], dtype=bool),
    )

    validation = cb._run_validation_gate_episodes(
        [{"name": "thin_window", "min_episodes": 1, "min_trades": 99}],
        {},
    )

    assert len(validation["thin_window"]) == 2


def test_effective_stage_is_used_for_recording_and_stage_timesteps() -> None:
    manager = _Manager()
    cb = _callback(manager)
    _quiet_step_methods(cb)
    cb._n_envs = 1
    cb._cur_rewards = [2.0]
    cb._cur_lens = [6]
    cb.num_timesteps = 100
    cb.locals = {
        "new_obs": np.zeros((1, 1), dtype=np.float32),
        "rewards": np.asarray([1.0]),
        "dones": np.asarray([True]),
        "infos": [{
            "terminal_info": {
                "episode_stats": {
                    "effective_curriculum_stage": "trend_student",
                    "trade_count": 2,
                    "profit_factor": 1.1,
                    "avg_r_multiple": 0.2,
                    "exit_quality_distribution": {},
                }
            }
        }],
    }

    assert cb._on_step() is True
    assert manager.recorded is not None
    assert manager.recorded["effective_stage"] is CurriculumStage.TREND_STUDENT
    assert cb._per_stage_stats["TREND_STUDENT"]["total_timesteps"] == 7
    comparison = cb._get_stage_comparison_data()
    assert comparison["stages"][0]["timesteps"] == 7


def test_invalid_nonempty_effective_stage_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid effective_curriculum_stage"):
        CurriculumTrainingCallback._resolve_effective_stage("made_up_stage")


def test_missing_effective_stage_and_accounting_failure_stop_training() -> None:
    manager = _Manager()
    cb = _callback(manager)
    _quiet_step_methods(cb)
    cb._n_envs = 1
    cb._cur_rewards = [0.0]
    cb._cur_lens = [0]
    cb.locals = {
        "new_obs": np.zeros((1, 1), dtype=np.float32),
        "rewards": np.asarray([0.0]),
        "dones": np.asarray([True]),
        "infos": [{"terminal_info": {"episode_stats": {}}}],
    }
    with pytest.raises(RuntimeError, match="omitted effective_curriculum_stage"):
        cb._on_step()

    def _broken_record(**_kwargs: Any) -> None:
        raise RuntimeError("accounting store unavailable")

    manager.record_episode_from_info = _broken_record  # type: ignore[method-assign]
    cb._cur_rewards = [0.0]
    cb._cur_lens = [0]
    cb.locals["infos"] = [{
        "terminal_info": {
            "episode_stats": {"effective_curriculum_stage": "INTEGRATOR"}
        }
    }]
    with pytest.raises(RuntimeError, match="accounting store unavailable"):
        cb._on_step()


def test_goal_stopping_honors_plateau_stop_setting() -> None:
    manager = _Manager()
    cb = _callback(manager, goal_based_stopping=True, plateau_stop=True)
    _quiet_step_methods(cb)
    cb._n_envs = 1
    cb._cur_rewards = [0.0]
    cb._cur_lens = [0]
    cb._ep_rewards.extend([0.0] * 10)
    cb.training_start_time = 123.0
    cb.num_timesteps = 100
    cb.locals = {
        "new_obs": np.zeros((1, 1), dtype=np.float32),
        "rewards": np.asarray([0.0]),
        "dones": np.asarray([False]),
        "infos": [{}],
    }

    assert cb._on_step() is True
    assert manager.stop_kwargs is not None
    assert manager.stop_kwargs["plateau_stop"] is True
