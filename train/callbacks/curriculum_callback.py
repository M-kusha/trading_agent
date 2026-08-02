
from __future__ import annotations

import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from train.obs_health import ObservationHealthTracker

try:
    from sb3_contrib.common.maskable.utils import get_action_masks as sb3_get_action_masks
    SB3_MASK_UTILS_AVAILABLE = True
except ImportError:
    sb3_get_action_masks = None  # type: ignore
    SB3_MASK_UTILS_AVAILABLE = False

from ..controllers import SmartClipController, SmartEntropyController, SmartLRController, TrainingHealthWatchdog

logger = logging.getLogger(__name__)


class CurriculumTrainingCallback(BaseCallback):

    def __init__(
        self,
        curriculum_manager: Any,
        total_timesteps: int,
        log_interval_steps: int = 50_000,
        save_path: str = "logs/curriculum",
        metrics_file: str = "logs/training/live_metrics.json",
        verbose: int = 1,
        enable_lr_warmup: bool = True,
        enable_checkpoints: bool = True,
        enable_entropy_schedule: bool = True,
        base_ent_coef: float = 0.02,

        enable_adaptive_clip_range: bool = True,
        enable_adaptive_lr: bool = True,
        base_clip_range: float = 0.2,

        goal_based_stopping: bool = False,
        max_hours: Optional[float] = None,
        plateau_stop: bool = True,
        plateau_threshold_episodes: int = 500,
        max_demotions_from_same_stage: int = 5,
        mastery_confirmation_episodes: int = 100,
    ):
        super().__init__(verbose)
        self.curriculum_manager = curriculum_manager
        self.total_timesteps = total_timesteps
        self.log_interval_steps = log_interval_steps
        self.save_path = Path(save_path)
        self.metrics_file = Path(metrics_file)

        # This callback writes live_metrics.json LAST in curriculum mode, so it
        # must publish observation health itself - otherwise it silently
        # overwrites the episode callback's payload and the panel disappears
        # from exactly the runs that matter most.
        self._obs_tracker = ObservationHealthTracker()
        self.enable_lr_warmup = enable_lr_warmup
        self.enable_checkpoints = enable_checkpoints
        self.enable_entropy_schedule = enable_entropy_schedule
        self.base_ent_coef = base_ent_coef


        self.enable_adaptive_clip_range = enable_adaptive_clip_range
        self.enable_adaptive_lr = enable_adaptive_lr
        self.base_clip_range = base_clip_range


        self.goal_based_stopping = goal_based_stopping
        self.max_hours = max_hours
        self.plateau_stop = plateau_stop
        self.plateau_threshold_episodes = plateau_threshold_episodes
        self.max_demotions_from_same_stage = max_demotions_from_same_stage
        self.mastery_confirmation_episodes = mastery_confirmation_episodes


        self._health_watchdog = TrainingHealthWatchdog(
            ev_critical_threshold=-0.5,
            ev_warning_threshold=0.05,
            ev_consecutive_failures=5,
            check_interval_steps=25_000,
            auto_stop=False,
        )

        self._last_log = 0
        self._last_metrics_save: float = 0.0
        self._n_envs = 1
        self._last_ent_update_step = 0
        self._last_clip_update_step = 0
        self._last_lr_update_step = 0


        MAX_EPISODE_HISTORY = 5000
        self._ep_rewards: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_pnls: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_win_rates: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_drawdowns: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_consecutive_losses: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_trades: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_lens: deque = deque(maxlen=MAX_EPISODE_HISTORY)


        self._cumulative_pnl: float = 0.0
        self._cumulative_trades: int = 0


        self._ep_profit_factors: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_r_multiples: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._ep_entry_quality: deque = deque(maxlen=MAX_EPISODE_HISTORY)
        self._exit_reason_counts: Dict[str, int] = {}


        self._reward_component_totals: Dict[str, float] = {}
        self._reward_component_counts: Dict[str, int] = {}


        self._latest_governor_state: Dict[str, float] = {}


        self._direction_stats: Dict[str, Any] = {
            "long_count": 0,
            "short_count": 0,
            "long_wins": 0,
            "short_wins": 0,
            "long_pnl": 0.0,
            "short_pnl": 0.0,
        }


        self._per_stage_stats: Dict[str, Dict[str, Any]] = {}
        self._current_stage_name: Optional[str] = None


        self._ppo_diagnostics: Dict[str, float] = {}
        self._n_updates: int = 0


        self._last_ingested_update: int = -1
        self._last_ingested_sig: Optional[tuple] = None

        self._cur_rewards: List[float] = []
        self._cur_lens: List[int] = []


        self._stage_history: deque = deque(maxlen=200)
        self._start_time: Optional[float] = None
        self._base_lr: Optional[float] = None
        self._current_lr: Optional[float] = None


        self._entropy_history: deque = deque(maxlen=20)
        self._kl_history: deque = deque(maxlen=20)
        self._clip_fraction_history: deque = deque(maxlen=20)
        self._policy_loss_history: deque = deque(maxlen=20)
        self._value_loss_history: deque = deque(maxlen=20)
        self._reward_history: deque = deque(maxlen=50)


        self._smart_entropy_controller = SmartEntropyController(initial_ent_coef=base_ent_coef)
        self._smart_lr_controller: Optional[SmartLRController] = None
        self._smart_clip_controller = SmartClipController(base_clip_range=base_clip_range)


        self._eval_env: Optional[Any] = None


        if self.curriculum_manager is not None:
            self.curriculum_manager.on_transition_callback = self._on_stage_transition

    @property
    def episodes_done(self) -> int:
        return len(self._ep_rewards)

    def _save_controller_states(self, path: Path) -> None:
        import json
        state = {
            "entropy_controller": self._smart_entropy_controller.to_dict() if self._smart_entropy_controller else None,
            "lr_controller": self._smart_lr_controller.to_dict() if self._smart_lr_controller else None,
            "clip_controller": self._smart_clip_controller.to_dict() if self._smart_clip_controller else None,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Controller states saved to {path}")

    def load_controller_states(self, path: Path) -> None:
        import json
        if not path.exists():
            logger.warning(f"Controller states file not found: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if state.get("entropy_controller") and self._smart_entropy_controller:
                self._smart_entropy_controller.load_from_dict(state["entropy_controller"])
            if state.get("lr_controller") and self._smart_lr_controller:
                self._smart_lr_controller.load_from_dict(state["lr_controller"])
            if state.get("clip_controller") and self._smart_clip_controller:
                self._smart_clip_controller.load_from_dict(state["clip_controller"])
            logger.info(f"Controller states loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load controller states: {e}")

    def _save_metrics_state(self, path: Path) -> None:
        import json
        state = {

            "ep_rewards": list(self._ep_rewards),
            "ep_pnls": list(self._ep_pnls),
            "ep_win_rates": list(self._ep_win_rates),
            "ep_drawdowns": list(self._ep_drawdowns),
            "ep_trades": list(self._ep_trades),
            "ep_lens": list(self._ep_lens),
            "ep_profit_factors": list(self._ep_profit_factors),
            "ep_r_multiples": list(self._ep_r_multiples),
            "ep_entry_quality": list(self._ep_entry_quality),

            "cumulative_pnl": self._cumulative_pnl,
            "cumulative_trades": self._cumulative_trades,

            "exit_reason_counts": self._exit_reason_counts,
            "direction_stats": self._direction_stats,

            "reward_component_totals": self._reward_component_totals,
            "reward_component_counts": self._reward_component_counts,

            "stage_history": list(self._stage_history),

            "num_timesteps": self.num_timesteps,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)
        logger.debug(f"Metrics state saved to {path} ({len(self._ep_rewards)} episodes)")

    def load_metrics_state(self, path: Path) -> None:
        import json
        if not path.exists():
            logger.warning(f"Metrics state file not found: {path}")
            return
        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)


            MAX_EPISODE_HISTORY = 5000
            self._ep_rewards = deque(state.get("ep_rewards", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_pnls = deque(state.get("ep_pnls", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_win_rates = deque(state.get("ep_win_rates", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_drawdowns = deque(state.get("ep_drawdowns", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_trades = deque(state.get("ep_trades", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_lens = deque(state.get("ep_lens", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_profit_factors = deque(state.get("ep_profit_factors", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_r_multiples = deque(state.get("ep_r_multiples", []), maxlen=MAX_EPISODE_HISTORY)
            self._ep_entry_quality = deque(state.get("ep_entry_quality", []), maxlen=MAX_EPISODE_HISTORY)


            self._cumulative_pnl = state.get("cumulative_pnl", 0.0)
            self._cumulative_trades = state.get("cumulative_trades", 0)


            self._exit_reason_counts = state.get("exit_reason_counts", {})
            self._direction_stats = state.get("direction_stats", {
                "long_count": 0, "short_count": 0, "long_wins": 0, "short_wins": 0,
                "long_pnl": 0.0, "short_pnl": 0.0
            })


            self._reward_component_totals = state.get("reward_component_totals", {})
            self._reward_component_counts = state.get("reward_component_counts", {})


            self._stage_history = state.get("stage_history", [])

            logger.info(f"Metrics state loaded from {path} ({len(self._ep_rewards)} episodes)")
        except Exception as e:
            logger.warning(f"Failed to load metrics state: {e}")

    def _on_stage_transition(
        self,
        transition_type: str,
        old_stage: Any,
        new_stage: Any,
        info: Dict[str, Any],
    ) -> None:
        logger.info(f"🔄 Stage transition: {transition_type} ({old_stage.name} → {new_stage.name})")


        new_stage_value = int(new_stage.value)
        self._smart_entropy_controller.on_stage_change(new_stage_value)
        if self._smart_lr_controller is not None:
            self._smart_lr_controller.on_stage_change(new_stage_value)
        self._smart_clip_controller.on_stage_change(new_stage_value)
        logger.info(f"  🎛️ PID controllers synced to stage {new_stage_value}")


        if self.enable_checkpoints and self.model is not None:
            try:
                checkpoint_dir = self.save_path / "stage_checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)


                model_path = checkpoint_dir / f"model_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.zip"
                self.model.save(str(model_path))


                state_path = checkpoint_dir / f"curriculum_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.json"
                self.curriculum_manager.save(state_path)


                controllers_path = checkpoint_dir / f"controllers_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.json"
                self._save_controller_states(controllers_path)


                metrics_path = checkpoint_dir / f"metrics_{old_stage.name}_to_{new_stage.name}_{self.num_timesteps}.json"
                self._save_metrics_state(metrics_path)

                logger.info(f"  📁 Checkpoint saved: {model_path.name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Checkpoint failed: {e}")

    def _on_training_start(self) -> None:
        env = self.training_env
        self._n_envs = int(getattr(env, "num_envs", 1))
        self._cur_rewards = [0.0] * self._n_envs
        self._cur_lens = [0] * self._n_envs
        self._start_time = time.time()
        self.training_start_time = self._start_time

        self.save_path.mkdir(parents=True, exist_ok=True)


        if self.model is not None:
            self._base_lr = float(self.model.learning_rate) if not callable(self.model.learning_rate) else None
            self._current_lr = self._base_lr


            if self._base_lr is not None:
                self._smart_lr_controller = SmartLRController(base_lr=self._base_lr)

                if self.curriculum_manager is not None:
                    stage = int(self.curriculum_manager.current_stage.value)
                    self._smart_lr_controller.on_stage_change(stage)
                    self._smart_clip_controller.on_stage_change(stage)


        if self.curriculum_manager is not None:
            self._wire_promotion_evaluators()


        self._save_live_metrics()

    def _wire_promotion_evaluators(self) -> None:

        def validation_gate_evaluator(scenarios, training_stats):
            return self._run_validation_gate_episodes(scenarios, training_stats)

        def stress_test_evaluator(scenarios, baseline_stats):
            return self._run_stress_test_episodes(scenarios, baseline_stats)

        self.curriculum_manager.set_validation_gate_evaluator(validation_gate_evaluator)
        self.curriculum_manager.set_stress_test_evaluator(stress_test_evaluator)
        logger.info("Wired validation gate and stress test evaluators for promotion checks")

    def _run_validation_gate_episodes(
        self,
        scenarios: list,
        training_stats: dict,
        episodes_per_scenario: int = 10,
    ) -> dict:
        results: Dict[str, List[Dict[str, Any]]] = {}

        if self.model is None or self.training_env is None:
            logger.warning("Cannot run validation: model or env not available")
            return results


        eval_env = getattr(self, "_eval_env", None) or self.training_env


        total_budget = int(getattr(getattr(self.curriculum_manager.stage_config, "validation", None), "validation_episodes", 0) or 0) if self.curriculum_manager is not None else 0
        if total_budget <= 0:
            total_budget = int(episodes_per_scenario) * max(len(scenarios), 1)
        per_scenario_budget = int(np.ceil(total_budget / max(len(scenarios), 1)))


        def _predict_deterministic(obs_in):
            masks = None
            try:
                if SB3_MASK_UTILS_AVAILABLE and sb3_get_action_masks is not None:
                    masks = sb3_get_action_masks(eval_env)
            except Exception:
                masks = None

            try:
                return self.model.predict(obs_in, deterministic=True, action_masks=masks)  # type: ignore[arg-type]
            except TypeError:
                return self.model.predict(obs_in, deterministic=True)  # type: ignore[arg-type]


        def _unwrap_base_env(venv):
            base = venv
            try:
                while hasattr(base, "venv"):
                    base = base.venv
                if hasattr(base, "envs"):
                    envs_list = base.envs
                    if envs_list and len(envs_list) > 0:
                        base = envs_list[0]
                while hasattr(base, "env"):
                    base = base.env
            except Exception:
                return None
            return base

        base_env = _unwrap_base_env(eval_env)

        logger.info(f"Running validation gate: {len(scenarios)} scenarios (budget ~{per_scenario_budget} eps each)")

        for scenario in scenarios:
            scenario_name = str(scenario.get("name", "unknown"))
            scenario_episodes: List[Dict[str, Any]] = []

            spread_mult = float(scenario.get("spread_multiplier", 1.0) or 1.0)
            slippage_mult = float(scenario.get("slippage_multiplier", 1.0) or 1.0)
            vol_filter = (scenario.get("volatility_filter", None) or None)
            trend_filter = (scenario.get("trend_filter", None) or None)
            min_eps = int(scenario.get("min_episodes", 0) or 0)
            min_trades = int(scenario.get("min_trades", 0) or 0)


            target_episodes = max(int(per_scenario_budget), int(min_eps), 1)
            max_episodes = max(target_episodes + 10, int(np.ceil(target_episodes * 1.5)))

            total_trades_collected = 0
            step_count = 0


            original_difficulty = getattr(base_env, "_data_difficulty", None) if base_env is not None else None

            try:

                if base_env is not None and hasattr(base_env, "set_data_difficulty"):
                    try:
                        from dataclasses import replace

                        from envs.curriculum.config.execution import DataDifficulty

                        stage_diff = getattr(self.curriculum_manager.stage_config, "data_difficulty", None) if self.curriculum_manager is not None else None
                        scenario_diff = replace(stage_diff) if stage_diff is not None else DataDifficulty()


                        scenario_diff.prefer_recent_data = False
                        scenario_diff.recent_data_weight = 1.0

                        if isinstance(vol_filter, str):
                            vf = vol_filter.lower().strip()
                            if vf == "high":
                                scenario_diff.volatility_percentile_range = (0.70, 1.0)
                            elif vf == "low":
                                scenario_diff.volatility_percentile_range = (0.0, 0.30)

                        if isinstance(trend_filter, str):
                            tf = trend_filter.lower().strip()
                            if tf in {"trend", "trending", "strong_trend"}:
                                scenario_diff.min_trend_clarity = max(float(scenario_diff.min_trend_clarity), 0.60)
                                scenario_diff.max_trend_clarity = 1.0
                            elif tf in {"range", "ranging", "choppy"}:
                                scenario_diff.min_trend_clarity = 0.0
                                scenario_diff.max_trend_clarity = min(float(getattr(scenario_diff, "max_trend_clarity", 1.0)), 0.35)

                        base_env.set_data_difficulty(scenario_diff)
                    except Exception as e:
                        logger.debug(f"Validation scenario '{scenario_name}': could not set data difficulty: {e}")


                stress_applied = self._apply_stress_to_env(eval_env, spread_mult, slippage_mult, 0)


                obs_result = eval_env.reset()
                obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result


                max_steps_per_ep = int(getattr(getattr(base_env, "config", None), "max_steps_per_episode", 3000) or 3000)
                max_steps = int(max_episodes) * max(3000, max_steps_per_ep + 5)

                while step_count < max_steps:
                    if len(scenario_episodes) >= max_episodes:
                        break
                    if len(scenario_episodes) >= target_episodes and total_trades_collected >= min_trades:
                        break

                    step_count += 1

                    action, _ = _predict_deterministic(obs)
                    obs, reward, done, info = eval_env.step(action)

                    done_flag = done[0] if hasattr(done, "__len__") else done
                    info_raw = info[0] if isinstance(info, list) and len(info) > 0 else info
                    info_dict: Dict[str, Any] = info_raw if isinstance(info_raw, dict) else {}

                    if done_flag:
                        ep_stats: Dict[str, Any] = info_dict.get("episode_stats", info_dict)
                        ep_info = info_dict.get("episode", {})
                        ep_reward = ep_info.get("r", 0.0) if isinstance(ep_info, dict) else 0.0

                        trades = int(ep_stats.get("trade_count", 0) or 0)
                        total_trades_collected += max(trades, 0)

                        scenario_episodes.append({
                            "win_rate": float(ep_stats.get("win_rate", 0.0)),
                            "profit_factor": float(ep_stats.get("profit_factor", 0.0)),
                            "total_pnl": float(ep_stats.get("total_pnl", 0.0)),
                            "avg_r_multiple": float(ep_stats.get("avg_r_multiple", 0.0)),
                            "trade_count": trades,
                            "max_drawdown": float(ep_stats.get("max_drawdown", 0.0)),
                            "dd_breach": bool(ep_stats.get("dd_breach", False)),
                            "episode_reward": float(ep_reward),
                            "stress_applied": bool(stress_applied),
                        })

                if step_count >= max_steps:
                    logger.warning(f"Validation scenario '{scenario_name}' hit step limit ({max_steps})")
                if min_trades > 0 and total_trades_collected < min_trades:
                    logger.warning(
                        f"Validation scenario '{scenario_name}' collected {total_trades_collected} trades "
                        f"(< {min_trades}) in {len(scenario_episodes)} episodes"
                    )

            except Exception as e:
                logger.warning(f"Validation scenario '{scenario_name}' failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
            finally:

                try:
                    self._restore_env_execution(eval_env)
                except Exception:
                    pass
                try:
                    if base_env is not None and hasattr(base_env, "set_data_difficulty"):
                        base_env.set_data_difficulty(original_difficulty)
                except Exception:
                    pass

            results[scenario_name] = scenario_episodes
            logger.info(f"Validation scenario '{scenario_name}': collected {len(scenario_episodes)} episodes")

        return results

    def _run_stress_test_episodes(
        self,
        scenarios: list,
        baseline_stats: dict,
        episodes_per_scenario: int = 5,
    ) -> list:
        results = []

        if self.model is None or self.training_env is None:
            logger.warning("Cannot run stress test: model or env not available")
            return results

        eval_env = getattr(self, '_eval_env', None) or self.training_env

        logger.info(f"Running stress test: {len(scenarios)} scenarios, {episodes_per_scenario} episodes each")

        for scenario in scenarios:
            scenario_name = scenario.get('name', 'unknown')
            scenario_episodes = []


            spread_mult = float(scenario.get('spread_multiplier', 1.0))
            slippage_mult = float(scenario.get('slippage_multiplier', 1.0))
            latency_add = int(scenario.get('latency_bars', 0))

            try:

                stress_applied = self._apply_stress_to_env(
                    eval_env, spread_mult, slippage_mult, latency_add
                )

                obs_result = eval_env.reset()
                obs = obs_result[0] if isinstance(obs_result, tuple) else obs_result

                episodes_collected = 0
                max_steps = episodes_per_scenario * 3000
                step_count = 0

                while episodes_collected < episodes_per_scenario and step_count < max_steps:
                    step_count += 1

                    masks = None
                    try:
                        if SB3_MASK_UTILS_AVAILABLE and sb3_get_action_masks is not None:
                            masks = sb3_get_action_masks(eval_env)
                    except Exception:
                        masks = None
                    try:
                        action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)  # type: ignore[arg-type]
                    except TypeError:
                        action, _ = self.model.predict(obs, deterministic=True)  # type: ignore[arg-type]
                    obs, reward, done, info = eval_env.step(action)


                    done_flag = done[0] if hasattr(done, '__len__') else done
                    info_raw = info[0] if isinstance(info, list) and len(info) > 0 else info
                    info_dict: Dict[str, Any] = info_raw if isinstance(info_raw, dict) else {}


                    if done_flag:
                        ep_stats: Dict[str, Any] = info_dict.get('episode_stats', info_dict)
                        scenario_episodes.append({
                            'win_rate': float(ep_stats.get('win_rate', 0.0)),
                            'profit_factor': float(ep_stats.get('profit_factor', 0.0)),
                            'total_pnl': float(ep_stats.get('total_pnl', 0.0)),
                            'avg_r_multiple': float(ep_stats.get('avg_r_multiple', 0.0)),
                            'trade_count': int(ep_stats.get('trade_count', 0)),
                            'max_drawdown': float(ep_stats.get('max_drawdown', 0.0)),
                            'stress_applied': stress_applied,
                        })
                        episodes_collected += 1


                if step_count >= max_steps:
                    logger.warning(f"Stress test scenario '{scenario_name}' hit step limit ({max_steps})")


                self._restore_env_execution(eval_env)

            except Exception as e:
                logger.warning(f"Stress test scenario '{scenario_name}' failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                self._restore_env_execution(eval_env)

            results.append(scenario_episodes)
            logger.info(f"Stress test scenario '{scenario_name}': collected {len(scenario_episodes)} episodes")

        return results

    def _apply_stress_to_env(
        self,
        env,
        spread_mult: float,
        slippage_mult: float,
        latency_add: int
    ) -> bool:
        try:

            base_env = env
            while hasattr(base_env, 'venv'):
                base_env = base_env.venv
            if hasattr(base_env, 'envs') and base_env.envs:
                base_env = base_env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env


            if hasattr(base_env, 'set_scenario_execution_overrides') and callable(base_env.set_scenario_execution_overrides):
                if not hasattr(self, '_original_exec_params'):
                    self._original_exec_params = {}
                self._original_exec_params['scenario_spread_mult'] = getattr(base_env, '_scenario_spread_mult', 1.0)
                self._original_exec_params['scenario_slippage_mult'] = getattr(base_env, '_scenario_slippage_mult', 1.0)
                self._original_exec_params['scenario_latency_add'] = getattr(base_env, '_scenario_latency_add', 0)

                base_env.set_scenario_execution_overrides(
                    spread_mult=float(spread_mult),
                    slippage_mult=float(slippage_mult),
                    latency_add=int(latency_add),
                )
                return True


            if not hasattr(base_env, '_exec') or base_env._exec is None:
                return False

            exec_model = base_env._exec

            if not hasattr(self, '_original_exec_params'):
                self._original_exec_params = {}
            self._original_exec_params['spread_mult'] = getattr(exec_model, '_spread_mult', 1.0)
            self._original_exec_params['slippage_mult'] = getattr(exec_model, '_slippage_mult', 1.0)
            if hasattr(exec_model, 'cfg'):
                self._original_exec_params['latency_bars'] = getattr(exec_model.cfg, 'latency_bars', 0)

            exec_model._spread_mult *= float(spread_mult)
            exec_model._slippage_mult *= float(slippage_mult)
            if hasattr(exec_model, 'cfg') and int(latency_add) > 0:
                exec_model.cfg.latency_bars += int(latency_add)

            return True

        except Exception as e:
            logger.debug(f"Could not apply stress to env: {e}")
            return False

    def _restore_env_execution(self, env) -> None:
        if not hasattr(self, '_original_exec_params') or not self._original_exec_params:
            return

        try:
            base_env = env
            while hasattr(base_env, 'venv'):
                base_env = base_env.venv
            if hasattr(base_env, 'envs') and base_env.envs:
                base_env = base_env.envs[0]
            while hasattr(base_env, 'env'):
                base_env = base_env.env


            if hasattr(base_env, 'set_scenario_execution_overrides') and callable(base_env.set_scenario_execution_overrides):
                base_env.set_scenario_execution_overrides(
                    spread_mult=float(self._original_exec_params.get('scenario_spread_mult', 1.0)),
                    slippage_mult=float(self._original_exec_params.get('scenario_slippage_mult', 1.0)),
                    latency_add=int(self._original_exec_params.get('scenario_latency_add', 0)),
                )
                self._original_exec_params = {}
                return

            if not hasattr(base_env, '_exec') or base_env._exec is None:
                return

            exec_model = base_env._exec

            if 'spread_mult' in self._original_exec_params:
                exec_model._spread_mult = self._original_exec_params['spread_mult']
            if 'slippage_mult' in self._original_exec_params:
                exec_model._slippage_mult = self._original_exec_params['slippage_mult']
            if 'latency_bars' in self._original_exec_params and hasattr(exec_model, 'cfg'):
                exec_model.cfg.latency_bars = self._original_exec_params['latency_bars']

            self._original_exec_params = {}

        except Exception as e:
            logger.debug(f"Could not restore env execution: {e}")

    def _on_rollout_start(self) -> None:
        self._update_ppo_diagnostics()

    def _on_rollout_end(self) -> None:
        self._update_ppo_diagnostics()
        self._maybe_save_live_metrics(force=True)

    def _maybe_save_live_metrics(self, force: bool = False) -> None:
        now = time.time()
        if force or (now - self._last_metrics_save >= 1.0):
            self._save_live_metrics_inline()
            self._last_metrics_save = now

    def _apply_lr_warmup(self) -> None:
        if not self.enable_lr_warmup or self.curriculum_manager is None or self.model is None:
            return

        if self._base_lr is None:
            return


        lr_mult = self.curriculum_manager.get_lr_multiplier()

        if lr_mult < 1.0:

            new_lr = self._base_lr * lr_mult


            if hasattr(self.model, 'lr_schedule'):
                self.model.lr_schedule = lambda _, lr=new_lr: lr
            elif hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = new_lr
        else:

            if hasattr(self.model, 'lr_schedule'):
                base = self._base_lr
                self.model.lr_schedule = lambda _, lr=base: lr
            elif hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = self._base_lr

    def _apply_entropy_schedule(self) -> None:
        if not self.enable_entropy_schedule or self.curriculum_manager is None or self.model is None:
            return


        current_entropy = self._entropy_history[-1] if self._entropy_history else 0.7
        current_ent_coef = float(getattr(self.model, 'ent_coef', self.base_ent_coef))
        stage = self.curriculum_manager.current_stage
        stage_value = int(stage.value)


        self._smart_entropy_controller.on_stage_change(stage_value)


        steps_elapsed = self.num_timesteps - self._last_ent_update_step
        if steps_elapsed < 5_000:
            return

        self._last_ent_update_step = self.num_timesteps


        n_valid_actions: Optional[int] = None
        try:
            if hasattr(self, 'training_env') and self.training_env is not None:
                venv = self.training_env


                if SB3_MASK_UTILS_AVAILABLE and sb3_get_action_masks is not None:
                    try:
                        masks = sb3_get_action_masks(venv)
                        if masks is not None and len(masks) > 0:
                            n_valid_actions = int(np.sum(masks[0]))
                    except Exception as e:
                        logger.debug(f"sb3_get_action_masks failed, falling back: {e}")


                if n_valid_actions is None:
                    current_env: Any = venv
                    while hasattr(current_env, 'venv'):
                        current_env = current_env.venv
                    if hasattr(current_env, 'envs'):
                        envs_list = current_env.envs
                        if envs_list and len(envs_list) > 0:
                            base_env: Any = envs_list[0]
                            while hasattr(base_env, 'env'):
                                if hasattr(base_env, 'action_masks') and callable(base_env.action_masks):
                                    mask = base_env.action_masks()
                                    n_valid_actions = int(np.asarray(mask, dtype=np.bool_).sum())
                                    break
                                base_env = base_env.env
                            if n_valid_actions is None and hasattr(base_env, 'action_masks'):
                                mask = base_env.action_masks()
                                n_valid_actions = int(np.asarray(mask, dtype=np.bool_).sum())
        except Exception as e:
            logger.debug(f"Could not get action masks for entropy normalization: {e}")


        new_ent_coef, reason, should_apply = self._smart_entropy_controller.get_ent_coef(
            current_entropy=current_entropy,
            current_ent_coef=current_ent_coef,
            timesteps_elapsed=steps_elapsed,
            n_valid_actions=n_valid_actions,
        )


        if self.verbose >= 1 and self.num_timesteps % 50_000 < self._n_envs:
            max_h = self._smart_entropy_controller._get_max_entropy(n_valid_actions)
            norm_current = current_entropy / max_h if max_h > 0 else 0
            norm_target = self._smart_entropy_controller.STAGE_TARGETS_NORMALIZED[stage_value]
            logger.info(
                f"🔍 Entropy Status: {reason} | should_apply={should_apply} | "
                f"entropy: {current_entropy:.3f} (norm: {norm_current:.2f}) | "
                f"target_norm: {norm_target:.2f} | ent_coef: {current_ent_coef:.4f}"
            )


        # ent_coef/clip_range/vf_coef are PPO fields, not BaseAlgorithm fields,
        # so self.model is not statically known to have them. The hasattr guard
        # is the real contract here; setattr states that plainly.
        if should_apply and hasattr(self.model, 'ent_coef'):
            setattr(self.model, 'ent_coef', new_ent_coef)
            if self.verbose >= 1:
                norm_target = self._smart_entropy_controller.STAGE_TARGETS_NORMALIZED[stage_value]
                max_h = self._smart_entropy_controller._get_max_entropy(n_valid_actions)
                raw_target = norm_target * max_h
                norm_current = current_entropy / max_h if max_h > 0 else 0
                logger.info(
                    f"🎛️ Entropy PID: {reason} | "
                    f"ent_coef: {current_ent_coef:.4f} → {new_ent_coef:.4f} | "
                    f"entropy: {current_entropy:.3f} (norm: {norm_current:.2f}) | "
                    f"target: {raw_target:.2f} (norm: {norm_target:.2f}) | "
                    f"valid_actions: {n_valid_actions or 'est'}"
                )

    def _apply_adaptive_clip_range(self) -> None:
        if not self.enable_adaptive_clip_range or self.model is None:
            return


        if self.curriculum_manager is not None:
            try:
                self._smart_clip_controller.on_stage_change(int(self.curriculum_manager.current_stage.value))
            except Exception:
                pass


        steps_elapsed = self.num_timesteps - self._last_clip_update_step
        if steps_elapsed < 15_000:
            return

        self._last_clip_update_step = self.num_timesteps


        progress = float(getattr(self.model, "_current_progress_remaining", 1.0))
        clip_range_attr = getattr(self.model, 'clip_range', self.base_clip_range)
        current_clip: float = self.base_clip_range
        if callable(clip_range_attr):
            try:
                raw_result = clip_range_attr(progress)
                if isinstance(raw_result, (int, float)):
                    current_clip = float(raw_result)
            except Exception:
                pass
        elif isinstance(clip_range_attr, (int, float)):
            current_clip = float(clip_range_attr)


        new_clip, reason, should_apply = self._smart_clip_controller.get_clip_range(
            current_clip=current_clip,
            timesteps_elapsed=steps_elapsed,
        )


        stage = self._smart_clip_controller.current_stage
        bounds = self._smart_clip_controller.STAGE_CLIP_BOUNDS.get(stage, (0.10, 0.30))


        if self.verbose >= 1 and self.num_timesteps % 50_000 < self._n_envs:
            logger.info(
                f"🔍 Clip PID [Stage {stage}]: {reason} | "
                f"clip: {current_clip:.3f} -> {new_clip:.3f} (bounds: {bounds[0]:.2f}-{bounds[1]:.2f})"
            )


        if should_apply and hasattr(self.model, 'clip_range'):
            def constant_clip_schedule(progress: float, val: float = new_clip) -> float:
                return val

            setattr(self.model, 'clip_range', constant_clip_schedule)
            if self.verbose >= 1:
                logger.info(
                    f"🎚️ Clip PID Applied [Stage {stage}]: {reason} | "
                    f"{current_clip:.3f} -> {new_clip:.3f}"
                )

    def _apply_adaptive_vf_coef(self) -> None:
        if self.model is None:
            return


        if self.num_timesteps - getattr(self, '_last_vf_update_step', 0) < 15_000:
            return

        self._last_vf_update_step = self.num_timesteps

        if len(self._value_loss_history) < 3:
            return

        avg_v_loss = sum(list(self._value_loss_history)[-10:]) / min(10, len(self._value_loss_history))
        current_vf_coef = float(getattr(self.model, 'vf_coef', 0.5))


        ev = self._ppo_diagnostics.get('explained_variance', 0.5)

        target_vf_coef = current_vf_coef
        adjustment_reason = "stable"


        if ev < 0:
            target_vf_coef = min(2.0, current_vf_coef * 1.5)
            adjustment_reason = f"CRITICAL_NEGATIVE_EV: {ev:.3f} - major vf_coef boost"

        elif avg_v_loss > 2.0:
            target_vf_coef = min(1.5, current_vf_coef * 1.25)
            adjustment_reason = f"HIGH_VALUE_LOSS: {avg_v_loss:.2f} - boosting vf_coef"
        elif avg_v_loss > 1.0:
            target_vf_coef = min(1.2, current_vf_coef * 1.1)
            adjustment_reason = f"ELEVATED_VALUE_LOSS: {avg_v_loss:.2f} - slight boost"
        elif ev < 0.15 and avg_v_loss > 0.3:

            target_vf_coef = min(1.2, current_vf_coef * 1.15)
            adjustment_reason = f"LOW_EV: {ev:.3f} with v_loss {avg_v_loss:.2f} - boost"
        elif avg_v_loss < 0.1 and ev > 0.3 and current_vf_coef > 0.5:

            target_vf_coef = max(0.5, current_vf_coef * 0.95)
            adjustment_reason = f"HEALTHY: v_loss={avg_v_loss:.2f}, EV={ev:.3f} - reducing vf_coef"


        if self.verbose >= 1 and self.num_timesteps % 50_000 < self._n_envs:
            logger.info(
                f"🔍 VF Coef Status: {adjustment_reason} | "
                f"vf_coef: {current_vf_coef:.3f} -> {target_vf_coef:.3f} | "
                f"avg_v_loss: {avg_v_loss:.3f}, EV: {ev:.3f}"
            )


        if hasattr(self.model, 'vf_coef') and abs(current_vf_coef - target_vf_coef) > 0.02:
            setattr(self.model, 'vf_coef', target_vf_coef)
            if self.verbose >= 1:
                logger.info(
                    f"🎛️ Adaptive vf_coef: {adjustment_reason} | "
                    f"{current_vf_coef:.3f} -> {target_vf_coef:.3f} | EV: {ev:.3f}"
                )

    def _apply_adaptive_learning_rate(self) -> None:
        if not self.enable_adaptive_lr or self.model is None or self._base_lr is None:
            return

        if self._smart_lr_controller is None:
            return


        if self.curriculum_manager is not None:
            try:
                if float(self.curriculum_manager.get_lr_multiplier()) < 1.0:
                    return
            except Exception:
                pass


        if self.curriculum_manager is not None:
            try:
                self._smart_lr_controller.on_stage_change(int(self.curriculum_manager.current_stage.value))
            except Exception:
                pass


        steps_elapsed = self.num_timesteps - self._last_lr_update_step
        if steps_elapsed < 40_000:
            return

        self._last_lr_update_step = self.num_timesteps


        current_lr = self._current_lr or self._base_lr
        try:
            if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                current_lr = float(self.model.policy.optimizer.param_groups[0]["lr"])
        except Exception:
            pass


        new_lr, reason, should_apply = self._smart_lr_controller.get_lr(
            current_lr=current_lr,
            timesteps_elapsed=steps_elapsed,
        )


        stage = self._smart_lr_controller.current_stage
        bounds = self._smart_lr_controller.STAGE_LR_MULTIPLIERS.get(stage, (0.3, 1.0))
        lr_bounds = (self._base_lr * bounds[0], self._base_lr * bounds[1])


        if self.verbose >= 1 and self.num_timesteps % 100_000 < self._n_envs:
            logger.info(
                f"🔍 LR PID [Stage {stage}]: {reason} | "
                f"lr: {current_lr:.2e} -> {new_lr:.2e} (bounds: {lr_bounds[0]:.2e}-{lr_bounds[1]:.2e})"
            )


        if should_apply:

            if hasattr(self.model, 'lr_schedule'):
                self.model.lr_schedule = lambda _, lr=new_lr: lr
            if hasattr(self.model, 'learning_rate'):
                self.model.learning_rate = new_lr


            try:
                if hasattr(self.model, "policy") and hasattr(self.model.policy, "optimizer"):
                    for g in self.model.policy.optimizer.param_groups:
                        g["lr"] = float(new_lr)
            except Exception:
                pass


            self._current_lr = new_lr

            if self.verbose >= 1:
                logger.info(
                    f"📊 LR PID Applied [Stage {stage}]: {reason} | "
                    f"{current_lr:.2e} -> {new_lr:.2e}"
                )

    def _on_step(self) -> bool:
        self._obs_tracker.observe(self.locals.get("new_obs"))
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True


        if self.curriculum_manager is not None:
            self.curriculum_manager.step_transition_state(timesteps=self._n_envs)
            self._apply_lr_warmup()
            self._apply_entropy_schedule()
            self._apply_adaptive_clip_range()
            self._apply_adaptive_vf_coef()
            self._apply_adaptive_learning_rate()

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.array(dones, dtype=np.bool_).reshape(-1)

        for i in range(min(self._n_envs, len(rewards_arr))):
            self._cur_rewards[i] += float(rewards_arr[i])
            self._cur_lens[i] += 1

        for i, done in enumerate(dones_arr[:self._n_envs]):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            ep_reward = self._cur_rewards[i]
            ep_len = self._cur_lens[i]

            finfo = info.get("terminal_info", info)


            ep_stats = finfo.get("episode_stats", info.get("episode_stats", {}))


            self._ep_rewards.append(ep_reward)
            self._ep_lens.append(ep_len)
            pnl = float(finfo.get("total_pnl", info.get("total_pnl", 0.0)))
            trades = int(finfo.get("trade_count", info.get("trade_count", 0)))
            self._ep_pnls.append(pnl)

            wr = float(finfo.get("win_rate", info.get("win_rate", 0.0)))
            if wr > 1.0:
                wr = wr / 100.0
            wr = float(np.clip(wr, 0.0, 1.0))
            self._ep_win_rates.append(wr)


            dd = float(finfo.get("drawdown", info.get("drawdown", 0.0)))
            if dd > 1.0:
                dd = dd / 100.0
            dd = float(np.clip(dd, 0.0, 1.0))
            self._ep_drawdowns.append(dd)
            self._ep_trades.append(trades)

            # The env tracks this per episode and curriculum_manager already
            # gates promotion on it. Recording it here too is what lets the
            # dashboard show a measured streak instead of a default.
            self._ep_consecutive_losses.append(
                int(ep_stats.get(
                    "max_consecutive_losses_reached",
                    finfo.get("consecutive_losses", info.get("consecutive_losses", 0)),
                ))
            )


            self._reward_history.append(ep_reward)
            if self._smart_lr_controller is not None:
                try:
                    self._smart_lr_controller.update_history(reward=ep_reward)
                except Exception:
                    pass


            self._cumulative_pnl += pnl
            self._cumulative_trades += trades


            self._ep_profit_factors.append(float(ep_stats.get("profit_factor", finfo.get("profit_factor", 0.0))))
            self._ep_r_multiples.append(float(ep_stats.get("avg_r_multiple", finfo.get("avg_r_multiple", 0.0))))
            self._ep_entry_quality.append(float(ep_stats.get("avg_entry_quality", finfo.get("avg_entry_quality", 0.5))))


            exit_dist = ep_stats.get("exit_quality_distribution", ep_stats.get("exit_distribution", {}))
            for reason, count in exit_dist.items():
                self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + int(count)


            reward_components = ep_stats.get("reward_components", {}) or {}
            for comp_name, comp_data in reward_components.items():
                if isinstance(comp_data, dict):
                    total = float(comp_data.get("total", 0.0))
                    count = int(comp_data.get("count", 0))
                else:
                    total = float(comp_data)
                    count = 1
                self._reward_component_totals[comp_name] = self._reward_component_totals.get(comp_name, 0.0) + total
                self._reward_component_counts[comp_name] = self._reward_component_counts.get(comp_name, 0) + count


            governor_state = ep_stats.get("governor_state", {}) or {}
            if governor_state:
                self._latest_governor_state = governor_state


            dir_stats = ep_stats.get("direction_stats", {})
            if dir_stats:
                self._direction_stats["long_count"] += int(dir_stats.get("long_count", 0))
                self._direction_stats["short_count"] += int(dir_stats.get("short_count", 0))

                long_cnt = int(dir_stats.get("long_count", 0))
                short_cnt = int(dir_stats.get("short_count", 0))
                long_wr = float(dir_stats.get("long_win_rate", 0))
                short_wr = float(dir_stats.get("short_win_rate", 0))

                if long_wr > 1.0:
                    long_wr /= 100.0
                if short_wr > 1.0:
                    short_wr /= 100.0
                long_wr = float(np.clip(long_wr, 0.0, 1.0))
                short_wr = float(np.clip(short_wr, 0.0, 1.0))
                self._direction_stats["long_wins"] += int(long_cnt * long_wr)
                self._direction_stats["short_wins"] += int(short_cnt * short_wr)
                self._direction_stats["long_pnl"] += float(dir_stats.get("long_pnl", 0))
                self._direction_stats["short_pnl"] += float(dir_stats.get("short_pnl", 0))


            if self.curriculum_manager is not None:
                stage_name = self.curriculum_manager.current_stage.name
            else:
                stage_name = finfo.get("curriculum_stage", info.get("curriculum_stage", "unknown"))
            self._record_stage_episode_stats(
                stage_name=stage_name,
                pnl=pnl,
                win_rate=wr,
                drawdown=float(finfo.get("drawdown", info.get("drawdown", 0.0))),
                trades=trades,
                profit_factor=float(ep_stats.get("profit_factor", finfo.get("profit_factor", 0.0))),
                r_multiple=float(ep_stats.get("avg_r_multiple", finfo.get("avg_r_multiple", 0.0))),
                reward=ep_reward,
                direction_stats=dir_stats,
            )


            if self.curriculum_manager is not None:
                try:
                    self.curriculum_manager.record_episode_from_info(
                        info=finfo,
                        episode_reward=ep_reward,
                        episode_length=ep_len,
                    )
                except Exception as e:
                    logger.debug(f"record_episode_from_info failed: {e}")


            if self.curriculum_manager is not None:
                current_stage = self.curriculum_manager.current_stage.name
            else:
                current_stage = finfo.get("curriculum_stage", info.get("curriculum_stage", "unknown"))
            if self._stage_history and self._stage_history[-1]["stage"] != current_stage:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": len(self._ep_rewards),
                })
            elif not self._stage_history:
                self._stage_history.append({
                    "stage": current_stage,
                    "timestep": self.num_timesteps,
                    "episode": 1,
                })


            self._cur_rewards[i] = 0.0
            self._cur_lens[i] = 0


            self._maybe_save_live_metrics(force=False)


        self._update_ppo_diagnostics()


        if hasattr(self, '_health_watchdog') and self._health_watchdog is not None:
            ev = self._ppo_diagnostics.get('explained_variance', 0.5)
            mean_reward = float(np.mean(list(self._ep_rewards)[-50:])) if self._ep_rewards else 0
            win_rate = float(np.mean(list(self._ep_win_rates)[-50:])) if self._ep_win_rates else 0.5
            entropy = float(self._entropy_history[-1]) if self._entropy_history else 0.7

            should_stop, reason = self._health_watchdog.check(
                explained_variance=ev,
                mean_reward=mean_reward,
                win_rate=win_rate / 100.0 if win_rate > 1 else win_rate,
                entropy=entropy,
                timestep=self.num_timesteps,
            )

            if should_stop:
                logger.error(f"⛔ TRAINING HALTED BY WATCHDOG: {reason}")
                self._save_live_metrics()
                return False


        if self.num_timesteps - self._last_log >= self.log_interval_steps:
            self._log_progress()
            self._last_log = self.num_timesteps


        self._maybe_save_live_metrics(force=False)


        if self.goal_based_stopping and self.curriculum_manager is not None:

            if self.episodes_done > 0 and self.episodes_done % 10 == 0:
                should_stop, reason = self.curriculum_manager.should_stop_training(
                    max_timesteps=None,
                    max_episodes=None,
                    max_hours=self.max_hours,
                    start_time=self.training_start_time,
                    plateau_stop=False,
                    plateau_threshold_episodes=self.plateau_threshold_episodes,
                    max_demotions_from_same_stage=self.max_demotions_from_same_stage,
                    mastery_confirmation_episodes=self.mastery_confirmation_episodes,
                )
                if should_stop:
                    logger.info(f"🎯 Goal-based stopping triggered: {reason}")
                    self._log_progress()
                    self._save_live_metrics()
                    return False

        return True

    def _update_ppo_diagnostics(self) -> None:
        if self.model is None:
            return


        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_obj = self.model.logger


                if hasattr(logger_obj, 'name_to_value'):
                    values = logger_obj.name_to_value


                    if self.num_timesteps % 50_000 < self._n_envs and values:
                        logger.debug(f"📊 SB3 logger keys: {list(values.keys())[:20]}")


                    self._ppo_diagnostics['policy_loss'] = float(values.get('train/policy_gradient_loss', values.get('train/policy_loss', 0)))
                    self._ppo_diagnostics['value_loss'] = float(values.get('train/value_loss', 0))
                    self._ppo_diagnostics['entropy'] = float(values.get('train/entropy', values.get('train/entropy_loss', 0)))
                    self._ppo_diagnostics['kl_divergence'] = float(values.get('train/approx_kl', 0))
                    self._ppo_diagnostics['clip_fraction'] = float(values.get('train/clip_fraction', 0))
                    self._ppo_diagnostics['explained_variance'] = float(values.get('train/explained_variance', 0))
                    self._ppo_diagnostics['learning_rate'] = float(values.get('train/learning_rate', 0))
                    self._ppo_diagnostics['clip_range'] = float(values.get('train/clip_range', 0.2))


                    n_updates_val = values.get('train/n_updates', 0)
                    try:
                        n_updates = int(n_updates_val) if n_updates_val is not None else 0
                    except Exception:
                        n_updates = 0
                    if n_updates > 0:
                        self._n_updates = n_updates


                    update_id = n_updates
                    if update_id <= 0:

                        try:
                            update_id = int(getattr(self.model, "_n_updates", 0) or 0)
                        except Exception:
                            update_id = 0

                    is_new_update = True
                    if update_id > 0:
                        is_new_update = (update_id != self._last_ingested_update)
                        if is_new_update:
                            self._last_ingested_update = update_id
                    else:

                        sig = (
                            round(float(self._ppo_diagnostics.get("policy_loss", 0) or 0), 10),
                            round(float(self._ppo_diagnostics.get("value_loss", 0) or 0), 10),
                            round(float(self._ppo_diagnostics.get("kl_divergence", 0) or 0), 10),
                            round(abs(float(self._ppo_diagnostics.get("entropy", 0) or 0)), 10),
                        )
                        is_new_update = (sig != self._last_ingested_sig)
                        if is_new_update:
                            self._last_ingested_sig = sig

                    if is_new_update:

                        if self.curriculum_manager is not None and self._ppo_diagnostics['entropy'] != 0:
                            entropy_val = abs(self._ppo_diagnostics['entropy'])
                            self.curriculum_manager.update_entropy(entropy_val)
                            self._entropy_history.append(entropy_val)


                        kl_val = float(self._ppo_diagnostics.get('kl_divergence', 0) or 0)
                        clip_frac = float(self._ppo_diagnostics.get('clip_fraction', 0) or 0)
                        if kl_val > 0:
                            self._kl_history.append(kl_val)
                            try:
                                self._smart_clip_controller.update_history(kl_divergence=kl_val)
                            except Exception:
                                pass
                        if clip_frac >= 0:

                            if np.isfinite(clip_frac):
                                self._clip_fraction_history.append(clip_frac)
                                try:
                                    self._smart_clip_controller.update_history(clip_fraction=clip_frac)
                                except Exception:
                                    pass


                        p_loss = abs(float(self._ppo_diagnostics.get('policy_loss', 0) or 0))
                        v_loss = abs(float(self._ppo_diagnostics.get('value_loss', 0) or 0))
                        if p_loss > 0 and np.isfinite(p_loss):
                            self._policy_loss_history.append(p_loss)
                            if self._smart_lr_controller is not None:
                                try:
                                    self._smart_lr_controller.update_history(policy_loss=p_loss)
                                except Exception:
                                    pass
                        if v_loss > 0 and np.isfinite(v_loss):
                            self._value_loss_history.append(v_loss)
                            if self._smart_lr_controller is not None:
                                try:
                                    self._smart_lr_controller.update_history(value_loss=v_loss)
                                except Exception:
                                    pass


            if self._n_updates == 0 and hasattr(self.model, '_n_updates'):
                self._n_updates = int(self.model._n_updates)


            if self._ppo_diagnostics.get('learning_rate', 0) == 0 and hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                if callable(lr):

                    progress = self.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                    lr = lr(1.0 - progress)
                self._ppo_diagnostics['learning_rate'] = float(lr)

        except Exception:
            pass

    def _record_stage_episode_stats(
        self,
        stage_name: str,
        pnl: float,
        win_rate: float,
        drawdown: float,
        trades: int,
        profit_factor: float,
        r_multiple: float,
        reward: float,
        direction_stats: Optional[Dict[str, Any]] = None,
    ) -> None:
        if stage_name not in self._per_stage_stats:
            self._per_stage_stats[stage_name] = {
                "stage_name": stage_name,
                "episodes": 0,
                "total_pnl": 0.0,
                "total_trades": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_drawdown": 0.0,
                "total_reward": 0.0,
                "sum_profit_factor": 0.0,
                "sum_r_multiple": 0.0,
                "pnl_values": [],
                "win_rate_values": [],
                "first_episode": len(self._ep_rewards),
                "first_timestep": self.num_timesteps,
                "last_episode": 0,
                "last_timestep": 0,

                "long_count": 0,
                "short_count": 0,
                "long_wins": 0,
                "short_wins": 0,
                "long_pnl": 0.0,
                "short_pnl": 0.0,
            }

        stats = self._per_stage_stats[stage_name]
        stats["episodes"] += 1
        stats["total_pnl"] += pnl
        stats["total_trades"] += trades
        stats["total_drawdown"] += drawdown
        stats["total_reward"] += reward
        stats["sum_profit_factor"] += profit_factor
        stats["sum_r_multiple"] += r_multiple
        stats["last_episode"] = len(self._ep_rewards)
        stats["last_timestep"] = self.num_timesteps


        if trades > 0:
            wins = int(round(win_rate * trades))
            losses = trades - wins
            stats["total_wins"] += wins
            stats["total_losses"] += losses


        stats["pnl_values"].append(pnl)
        stats["win_rate_values"].append(win_rate)
        if len(stats["pnl_values"]) > 100:
            stats["pnl_values"] = stats["pnl_values"][-100:]
        if len(stats["win_rate_values"]) > 100:
            stats["win_rate_values"] = stats["win_rate_values"][-100:]


        if direction_stats:
            stats["long_count"] += int(direction_stats.get("long_count", 0))
            stats["short_count"] += int(direction_stats.get("short_count", 0))
            long_cnt = int(direction_stats.get("long_count", 0))
            short_cnt = int(direction_stats.get("short_count", 0))
            long_wr = float(direction_stats.get("long_win_rate", 0))
            short_wr = float(direction_stats.get("short_win_rate", 0))

            if long_wr > 1.0:
                long_wr /= 100.0
            if short_wr > 1.0:
                short_wr /= 100.0
            long_wr = float(np.clip(long_wr, 0.0, 1.0))
            short_wr = float(np.clip(short_wr, 0.0, 1.0))
            stats["long_wins"] += int(long_cnt * long_wr)
            stats["short_wins"] += int(short_cnt * short_wr)
            stats["long_pnl"] += float(direction_stats.get("long_pnl", 0))
            stats["short_pnl"] += float(direction_stats.get("short_pnl", 0))

    def _get_stage_comparison_data(self) -> Dict[str, Any]:
        stage_order = [
            "EXPLORER", "EXPERIMENTER", "TREND_STUDENT", "SESSION_STUDENT",
            "TIMING_STUDENT", "INTEGRATOR", "RISK_MANAGER", "STRATEGIST",
            "PROFESSIONAL", "LIVE_READY"
        ]

        stages_data = []
        prev_stats = None

        for stage_name in stage_order:
            stats = self._per_stage_stats.get(stage_name)
            if stats is None or stats["episodes"] == 0:
                continue

            episodes = stats["episodes"]
            total_trades = stats["total_trades"]


            avg_pnl = stats["total_pnl"] / episodes if episodes > 0 else 0
            avg_trades = total_trades / episodes if episodes > 0 else 0
            avg_drawdown = stats["total_drawdown"] / episodes if episodes > 0 else 0
            avg_reward = stats["total_reward"] / episodes if episodes > 0 else 0
            avg_profit_factor = stats["sum_profit_factor"] / episodes if episodes > 0 else 0
            avg_r_multiple = stats["sum_r_multiple"] / episodes if episodes > 0 else 0


            total_trades_wl = stats["total_wins"] + stats["total_losses"]
            win_rate = stats["total_wins"] / total_trades_wl if total_trades_wl > 0 else 0


            pnl_std = float(np.std(stats["pnl_values"])) if len(stats["pnl_values"]) > 1 else 0
            win_rate_std = float(np.std(stats["win_rate_values"])) if len(stats["win_rate_values"]) > 1 else 0


            stage_entry = {
                "stage_name": stage_name,
                "stage_index": stage_order.index(stage_name),
                "episodes": episodes,
                "timesteps": stats["last_timestep"] - stats["first_timestep"],
                "total_trades": total_trades,
                "total_pnl": stats["total_pnl"],
                "avg_pnl": avg_pnl,
                "pnl_std": pnl_std,
                "win_rate": win_rate * 100,
                "win_rate_std": win_rate_std * 100,
                "avg_drawdown": avg_drawdown * 100,
                "avg_trades": avg_trades,
                "avg_reward": avg_reward,
                "avg_profit_factor": avg_profit_factor,
                "avg_r_multiple": avg_r_multiple,
                "first_episode": stats["first_episode"],
                "last_episode": stats["last_episode"],
            }


            long_cnt = stats.get("long_count", 0)
            short_cnt = stats.get("short_count", 0)
            total_dir = long_cnt + short_cnt
            stage_entry["direction_stats"] = {
                "long_count": long_cnt,
                "short_count": short_cnt,
                "long_wins": stats.get("long_wins", 0),
                "short_wins": stats.get("short_wins", 0),
                "long_pnl": stats.get("long_pnl", 0.0),
                "short_pnl": stats.get("short_pnl", 0.0),
                "long_win_rate": (stats.get("long_wins", 0) / long_cnt * 100) if long_cnt > 0 else 0,
                "short_win_rate": (stats.get("short_wins", 0) / short_cnt * 100) if short_cnt > 0 else 0,
                "long_pct": (long_cnt / total_dir * 100) if total_dir > 0 else 50,
                "short_pct": (short_cnt / total_dir * 100) if total_dir > 0 else 50,
                "direction_ratio": (long_cnt / short_cnt) if short_cnt > 0 else 1.0,
            }


            if prev_stats is not None:
                prev_win_rate = prev_stats.get("win_rate", 0)
                prev_pnl = prev_stats.get("avg_pnl", 0)
                prev_pf = prev_stats.get("avg_profit_factor", 0)
                prev_reward = prev_stats.get("avg_reward", 0)

                stage_entry["improvement"] = {
                    "win_rate_delta": (win_rate * 100) - prev_win_rate,
                    "pnl_delta": avg_pnl - prev_pnl,
                    "profit_factor_delta": avg_profit_factor - prev_pf,
                    "reward_delta": avg_reward - prev_reward,
                }
            else:
                stage_entry["improvement"] = None

            stages_data.append(stage_entry)
            prev_stats = stage_entry


        overall_improvement = {}
        if len(stages_data) >= 2:
            first = stages_data[0]
            last = stages_data[-1]
            overall_improvement = {
                "win_rate_delta": last["win_rate"] - first["win_rate"],
                "pnl_delta": last["avg_pnl"] - first["avg_pnl"],
                "profit_factor_delta": last["avg_profit_factor"] - first["avg_profit_factor"],
                "reward_delta": last["avg_reward"] - first["avg_reward"],
                "stages_progressed": last["stage_index"] - first["stage_index"],
            }

        return {
            "stages": stages_data,
            "overall_improvement": overall_improvement,
            "current_stage": self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A",
            "total_stages_visited": len(stages_data),
        }

    def _consecutive_loss_stats(self) -> Dict[str, Any]:
        """Losing-streak stats over the same 50-episode window episode_callback uses.

        A streak of 3+ is the threshold the streak rate counts, matching
        episode_callback so both writers of live_metrics.json mean the same
        thing by these keys.
        """
        window = list(self._ep_consecutive_losses)[-50:]
        if not window:
            return {
                "max_consecutive_losses": None,
                "avg_consecutive_losses": None,
                "consecutive_loss_streak_rate": None,
            }
        return {
            "max_consecutive_losses": int(max(window)),
            "avg_consecutive_losses": float(np.mean(window)),
            "consecutive_loss_streak_rate": float(sum(1 for x in window if x >= 3) / len(window)),
        }

    def _get_stage_config_version(self) -> Dict[str, Any]:
        if self.curriculum_manager is None:
            return {"stage": "N/A", "hash": "N/A"}

        try:
            import hashlib
            stage = self.curriculum_manager.current_stage

            config = self.curriculum_manager.stage_config


            key_params = {
                "stage": stage.name,
                "max_steps_per_episode": config.max_steps_per_episode,
                "trailing_stop_bonus": getattr(config.rewards, "trailing_stop_bonus", None),
                "agent_close_bonus": getattr(config.rewards, "agent_close_bonus", None),
                "good_loss_cut_bonus": getattr(config.rewards, "good_loss_cut_bonus", None),
                "daily_trade_soft_limit": getattr(config.rewards, "daily_trade_soft_limit", None),
                "entry_quality_threshold": getattr(config.constraints, "entry_quality_threshold", None),
                "min_profit_factor": getattr(config.competence, "min_profit_factor", None),
                "min_avg_pnl": getattr(config.competence, "min_avg_pnl", None),
                "max_consecutive_loss_rate": getattr(config.competence, "max_consecutive_loss_rate", None),
            }


            param_str = str(sorted(key_params.items()))
            config_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]

            return {
                "stage": stage.name,
                "hash": config_hash,
                "key_params": key_params,
            }
        except Exception as e:
            return {"stage": "ERROR", "hash": str(e)[:50]}

    def _get_curriculum_progress(self) -> Dict[str, Any]:
        if self.curriculum_manager is None:
            return {}

        try:

            report = self.curriculum_manager.get_progress_report()
            return report
        except Exception as e:
            logger.warning(f"Failed to get curriculum progress: {e}")
            return {
                "current_stage": self.curriculum_manager.current_stage.name,
                "stage_index": self.curriculum_manager.current_stage.value,
                "stage_episodes": self.curriculum_manager.stage_episodes,
                "stage_timesteps": self.curriculum_manager.stage_timesteps,
                "total_episodes": self.curriculum_manager.total_episodes,
                "total_timesteps": self.curriculum_manager.total_timesteps,
            }

    def _log_progress(self) -> None:
        if not self._ep_rewards:
            return


        n = min(50, len(self._ep_rewards))
        rewards = list(self._ep_rewards)[-n:]
        pnls = list(self._ep_pnls)[-n:]
        win_rates = list(self._ep_win_rates)[-n:]
        drawdowns = list(self._ep_drawdowns)[-n:]
        trades = list(self._ep_trades)[-n:]

        mean_reward = float(np.mean(rewards))
        mean_pnl = float(np.mean(pnls))
        mean_wr = float(np.mean(win_rates))
        max_dd = float(np.max(drawdowns))
        mean_trades = float(np.mean(trades))

        progress = 100.0 * (self.num_timesteps / max(self.total_timesteps, 1))

        stage = self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A"

        logger.info(
            f"Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
            f"Stage: {stage} | "
            f"Reward {mean_reward:+.3f} | PnL €{mean_pnl:+.0f} | "
            f"WR {mean_wr:.1%} | Trades {mean_trades:.1f} | MaxDD {max_dd:.1%}"
        )

    def _save_live_metrics(self) -> None:
        self._save_live_metrics_inline()

    def _save_live_metrics_inline(self) -> None:
        try:
            metrics_file = self.metrics_file
            metrics_file.parent.mkdir(parents=True, exist_ok=True)

            n_recent = min(100, len(self._ep_rewards))


            fps = 0.0
            if self._start_time is not None and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                if elapsed > 0:
                    fps = self.num_timesteps / elapsed


            eta_seconds = 0.0
            if self._start_time and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                remaining = self.total_timesteps - self.num_timesteps
                rate = self.num_timesteps / max(elapsed, 1)
                eta_seconds = remaining / max(rate, 1)


            mean_reward = float(np.mean(list(self._ep_rewards)[-50:])) if self._ep_rewards else 0.0
            mean_pnl = float(np.mean(list(self._ep_pnls)[-50:])) if self._ep_pnls else 0.0
            total_pnl = self._cumulative_pnl
            mean_win_rate = float(np.mean(list(self._ep_win_rates)[-50:])) if self._ep_win_rates else 0.0
            max_drawdown = float(np.max(list(self._ep_drawdowns)[-50:])) if self._ep_drawdowns else 0.0
            mean_trades = float(np.mean(list(self._ep_trades)[-50:])) if self._ep_trades else 0.0
            total_trades = self._cumulative_trades


            mean_profit_factor = float(np.mean(list(self._ep_profit_factors)[-50:])) if self._ep_profit_factors else 0.0
            mean_r_multiple = float(np.mean(list(self._ep_r_multiples)[-50:])) if self._ep_r_multiples else 0.0
            mean_entry_quality = float(np.mean(list(self._ep_entry_quality)[-50:])) if self._ep_entry_quality else 0.5


            reward_component_avgs = {}
            for comp_name, total in self._reward_component_totals.items():
                count = self._reward_component_counts.get(comp_name, 1)
                reward_component_avgs[comp_name] = total / max(count, 1)

            metrics = {
                "timestamp": datetime.now().isoformat(),
                "progress": {
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                    "total_episodes": len(self._ep_rewards),
                    "eta_seconds": eta_seconds,
                },
                "observation": self._obs_tracker.report(),

                "learning": {
                    "fps": fps,
                    "n_updates": self._n_updates,
                    "mean_reward": mean_reward,
                    "total_pnl": total_pnl,
                    "policy_loss": self._ppo_diagnostics.get('policy_loss', 0),
                    "value_loss": self._ppo_diagnostics.get('value_loss', 0),


                    "entropy": abs(self._ppo_diagnostics.get('entropy', 0)),
                    "kl_divergence": self._ppo_diagnostics.get('kl_divergence', 0),
                    "clip_fraction": self._ppo_diagnostics.get('clip_fraction', 0),
                    "explained_variance": self._ppo_diagnostics.get('explained_variance', 0),
                    "learning_rate": self._ppo_diagnostics.get('learning_rate', self._current_lr or 0),
                },
                "trading": {
                    "total_trades": total_trades,
                    "mean_trades": mean_trades,
                    "mean_win_rate": mean_win_rate * 100,

                    "max_drawdown": max_drawdown,
                },

                "direction_stats": {
                    "long_count": self._direction_stats["long_count"],
                    "short_count": self._direction_stats["short_count"],
                    "long_wins": self._direction_stats["long_wins"],
                    "short_wins": self._direction_stats["short_wins"],
                    "long_pnl": self._direction_stats["long_pnl"],
                    "short_pnl": self._direction_stats["short_pnl"],
                    "long_win_rate": (self._direction_stats["long_wins"] / max(self._direction_stats["long_count"], 1)) * 100,
                    "short_win_rate": (self._direction_stats["short_wins"] / max(self._direction_stats["short_count"], 1)) * 100,
                    "direction_ratio": self._direction_stats["long_count"] / max(self._direction_stats["short_count"], 1),
                },

                # Without the consecutive-loss fields the dashboard fell back to
                # q.get(key, 0), so an unmeasured streak rendered as a flawless
                # zero - the same way absence read as health during the blind
                # observation outage. Definitions match episode_callback so the
                # panel means one thing regardless of which callback wrote last.
                "quality": {
                    "mean_profit_factor": mean_profit_factor,
                    "mean_r_multiple": mean_r_multiple,
                    "mean_entry_quality": mean_entry_quality,
                    **self._consecutive_loss_stats(),
                },

                "exit_stats": {"distribution": dict(self._exit_reason_counts)},

                "reward_components": reward_component_avgs,

                "reward_component_totals": dict(self._reward_component_totals),
                "reward_component_counts": dict(self._reward_component_counts),
                "curriculum_stage": self.curriculum_manager.current_stage.name if self.curriculum_manager else "N/A",

                "stage_config_version": self._get_stage_config_version(),

                "curriculum_progress": self._get_curriculum_progress(),
                "recent_rewards": [float(x) for x in list(self._ep_rewards)[-n_recent:]],
                "recent_pnls": [float(x) for x in list(self._ep_pnls)[-n_recent:]],

                "recent_win_rates": [float(x) for x in list(self._ep_win_rates)[-n_recent:]],
                "recent_drawdowns": [float(x) for x in list(self._ep_drawdowns)[-n_recent:]],
                "recent_trades": [int(x) for x in list(self._ep_trades)[-n_recent:]],
                "recent_profit_factors": [float(x) for x in list(self._ep_profit_factors)[-n_recent:]],
                "recent_r_multiples": [float(x) for x in list(self._ep_r_multiples)[-n_recent:]],
                "stage_history": list(self._stage_history),
                "stage_comparison": self._get_stage_comparison_data(),

                "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "mean_reward": mean_reward,
                "mean_pnl": mean_pnl,
                "total_pnl": total_pnl,
                "mean_win_rate": mean_win_rate,
                "max_drawdown": max_drawdown,

                "governor": self._latest_governor_state or {
                    "loss_layer_ratio": 0.0,
                    "loss_layer_level": 0.0,
                    "win_streak_ratio": 0.0,
                    "session_pnl_headroom": 1.0,
                    "session_trade_budget": 1.0,
                    "session_consec_loss_ratio": 0.0,
                    "session_progress": 0.0,
                    "pending_order_progress": 0.0,
                },
            }


            tmp_file = metrics_file.with_suffix('.json.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f)

            try:
                tmp_file.replace(metrics_file)
            except PermissionError:
                import shutil
                try:
                    shutil.copy2(tmp_file, metrics_file)
                    tmp_file.unlink(missing_ok=True)
                except Exception:
                    with open(metrics_file, 'w', encoding='utf-8') as f:
                        json.dump(metrics, f)
                    tmp_file.unlink(missing_ok=True)

        except Exception as e:
            import traceback
            logger.error(f"Failed to save live metrics: {e}")
            traceback.print_exc()
