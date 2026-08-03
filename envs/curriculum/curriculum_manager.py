

from __future__ import annotations

import copy
import json
import math
from collections import deque
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from envs.core.shared_utils import (
    clamp as _cl,
)
from envs.core.shared_utils import (
    get_envs_logger,
    iso_timestamp,
)
from envs.core.shared_utils import (
    mean_ci_normal as _mci,
)
from envs.core.shared_utils import (
    safe_float as _sf,
)
from envs.core.shared_utils import (
    safe_int as _si,
)
from envs.core.shared_utils import (
    wilson_interval as _wi,
)
from envs.curriculum.curriculum_config import (
    MIN_EVALUATION_EPISODES,
    CompetenceThresholds,
    CurriculumStage,
    CurriculumStageConfig,
    get_next_stage,
    get_previous_stage,
    get_stage_config,
    get_stage_progression,
)
from envs.curriculum.curriculum_invariants import (
    AntiGamingChecker,
    CurriculumInvariantChecker,
)
from envs.curriculum.metrics import (
    CompositeScore,
    EpisodeMetrics,
    LearningVelocity,
    RollingStats,
    compute_adjusted_thresholds,
    compute_composite_score,
)
from envs.curriculum.protocols import (
    RecoveryProtocolState,
    ReviewSessionState,
)
from envs.curriculum.regime_skill_assessment import (
    RegimeSkillAssessment,
    TradeWithRegime,
)
from envs.curriculum.skills import (
    DemotionAnalyzer,
    SkillAssessment,
)
from envs.curriculum.validation_gates import (
    StressTestRunner,
    ValidationGateChecker,
)

logger = get_envs_logger("curriculum_manager")

# Below this many trades in the evaluation window, a win rate or profit factor
# is not a measurement. Used to decide whether a low-activity policy has
# earned the right to have its trade-count floor waived.
MIN_TRADES_FOR_VALID_RATE = 30

STATE_VERSION = "2.2"
DEFAULT_TZ = "Europe/Berlin"


DEMOTION_HISTORY_LIMIT = 200
PROCESSED_EPISODE_ID_LIMIT = 5000
MIN_TRADES_FOR_WILSON_GATE_DEFAULT = 30


def _now_iso(tz: str = DEFAULT_TZ) -> str:
    try:
        return datetime.now(tz=ZoneInfo(tz)).isoformat()
    except Exception as e:
        logger.debug(f"Timezone fallback for {tz}: {e}")
        return iso_timestamp()


def _safe_float(x: Any, default: float = 0.0) -> float:
    return _sf(x, default)


def _safe_int(x: Any, default: int = 0) -> int:
    return _si(x, default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return _cl(v, lo, hi)


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    return _wi(k, n, z)


def _mean_ci_normal(mean: float, std: float, n: int, z: float = 1.96) -> Tuple[float, float]:
    return _mci(mean, std, n, z)


def _get_progression() -> List[CurriculumStage]:
    return get_stage_progression()


def _stage_to_index(stage: CurriculumStage) -> int:
    prog = _get_progression()
    try:
        return prog.index(stage)
    except ValueError:
        return 0


def _index_to_stage(idx: int) -> CurriculumStage:
    prog = _get_progression()
    idx = max(0, min(idx, len(prog) - 1))
    return prog[idx]


def _foundation_stage() -> CurriculumStage:
    return _get_progression()[0]


def _linear_regression_slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    try:
        coeffs = np.polyfit(x, y, 1)
        return float(coeffs[0])
    except Exception as e:
        logger.debug(f"Linear regression failed: {e}")
        return 0.0


class CurriculumManager:

    def __init__(
        self,
        initial_stage: Optional[CurriculumStage] = None,
        max_history_size: int = 1000,
        auto_promote: bool = True,
        auto_demote: bool = True,
        verbose: bool = True,
        tz: str = DEFAULT_TZ,
        on_transition_callback: Optional[Callable] = None,
        rng_seed: Optional[int] = None,
        validation_evaluator: Optional[Callable[[CurriculumStageConfig], Dict[str, Any]]] = None,
        stress_test_evaluator: Optional[Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]] = None,
        validation_gate_evaluator: Optional[Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]] = None,
        bars_per_trading_day: int = 96,
    ) -> None:

        if initial_stage is None:
            initial_stage = _foundation_stage()
        self.current_stage: CurriculumStage = initial_stage
        self.max_history_size = max_history_size
        self.auto_promote = auto_promote
        self.auto_demote = auto_demote
        self.verbose = verbose
        self.tz = tz
        self.on_transition_callback = on_transition_callback
        self.validation_evaluator = validation_evaluator
        self.stress_test_evaluator = stress_test_evaluator
        self.validation_gate_evaluator = validation_gate_evaluator
        self.bars_per_trading_day = bars_per_trading_day


        self._rng = np.random.default_rng(rng_seed)


        self._progression = get_stage_progression()


        self._history: Dict[CurriculumStage, Deque[EpisodeMetrics]] = {
            stage: deque(maxlen=max_history_size) for stage in self._progression
        }


        self._stage_timesteps_total: Dict[CurriculumStage, int] = {stage: 0 for stage in self._progression}
        self._stage_episodes_total: Dict[CurriculumStage, int] = {stage: 0 for stage in self._progression}


        self.stage_timesteps: int = 0
        self.stage_episodes: int = 0


        self._stage_epoch_counter: Dict[CurriculumStage, int] = {stage: 0 for stage in self._progression}
        self._current_stage_epoch: int = 0


        self._transitions: List[Dict[str, Any]] = []


        self._rolling_stats: Optional[RollingStats] = None
        self._rolling_stats_dirty: bool = True


        self.total_timesteps: int = 0
        self.total_episodes: int = 0


        self._transition_cooldown_remaining: int = 0
        self._reward_blend_remaining: int = 0
        self._previous_stage_config: Optional[CurriculumStageConfig] = None

        self._previous_stage_name: Optional[str] = None
        self._lr_warmup_active: bool = False
        self._lr_warmup_steps_remaining: int = 0
        self._lr_warmup_factor: float = 1.0


        self._learning_velocity = LearningVelocity()
        self._skill_assessment: Optional[SkillAssessment] = None
        self._demotion_analyzer = DemotionAnalyzer()
        self._recovery_state = RecoveryProtocolState()
        self._review_state = ReviewSessionState()
        self._composite_score: Optional[CompositeScore] = None


        self._selectivity_phase_active: bool = False
        self._selectivity_phase_episodes_remaining: int = 0
        self._selectivity_phase_requirements: Dict[str, Any] = {}
        self._selectivity_phase_completed_epoch: int = -1
        self._selectivity_phase_target_episodes: int = 0
        self._selectivity_phase_failures: int = 0

        # Promotion validation is intentionally re-run after the policy has
        # changed, but not after every single episode.  The tuple is scoped to
        # a stage epoch so a demotion/re-entry can never inherit a stale retry
        # delay from an earlier visit to the same stage.
        self._last_validation_attempt_stage: Optional[str] = None
        self._last_validation_attempt_epoch: int = -1
        self._last_validation_attempt_stage_episode: int = -1


        self._invariant_checker = CurriculumInvariantChecker(verbose=verbose)
        self._anti_gaming_checker = AntiGamingChecker()
        self._validation_gate = ValidationGateChecker()
        self._stress_tester = StressTestRunner()
        self._regime_assessment: Optional[RegimeSkillAssessment] = None


        self._regime_trades: Deque[TradeWithRegime] = deque(maxlen=2000)
        self._regime_assessment_dirty: bool = True


        self._validation_gate_history: List[Dict[str, Any]] = []


        self._stress_test_history: List[Dict[str, Any]] = []


        self._last_episode_end_idx: int = -1


        self._processed_episode_ids: Deque[int] = deque()
        self._processed_episode_id_set: Set[int] = set()


        self._review_tick_episode: int = -1


        self._cached_effective_stage: Optional[CurriculumStage] = None
        self._cached_effective_stage_episode: int = -1


        self._current_entropy: float = -1.0


        self._enter_stage(self.current_stage, reason="init")

        logger.info(f"CurriculumManager v{STATE_VERSION} initialized at stage: {initial_stage.name}")


    def set_stress_test_evaluator(
        self,
        evaluator: Callable[[List[Dict[str, Any]], Dict[str, Any]], List[Dict[str, Any]]]
    ) -> None:
        self.stress_test_evaluator = evaluator

    def set_validation_evaluator(
        self,
        evaluator: Callable[[CurriculumStageConfig], Dict[str, Any]]
    ) -> None:
        self.validation_evaluator = evaluator

    def set_validation_gate_evaluator(
        self,
        evaluator: Callable[[List[Dict[str, Any]], Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]
    ) -> None:
        self.validation_gate_evaluator = evaluator


    def _enter_stage(
        self,
        stage: CurriculumStage,
        reason: str,
        demoted_from_stage: Optional[CurriculumStage] = None,
    ) -> None:

        if hasattr(self, 'current_stage') and self.current_stage != stage:
            self._previous_stage_config = get_stage_config(self.current_stage)
            self._previous_stage_name = self.current_stage.name

        self._stage_epoch_counter[stage] += 1
        self._current_stage_epoch = self._stage_epoch_counter[stage]
        self.stage_timesteps = 0
        self.stage_episodes = 0
        self._rolling_stats_dirty = True


        new_config = get_stage_config(stage)
        transition = new_config.transition


        self._transition_cooldown_remaining = transition.transition_cooldown_episodes


        if transition.reward_blend_enabled and self._previous_stage_config is not None:
            self._reward_blend_remaining = transition.reward_blend_episodes
        else:
            self._reward_blend_remaining = 0


        if transition.lr_warmup_enabled and reason != "init":
            self._lr_warmup_active = True
            self._lr_warmup_steps_remaining = transition.lr_warmup_steps
            self._lr_warmup_factor = transition.lr_warmup_factor
        else:
            self._lr_warmup_active = False
            self._lr_warmup_steps_remaining = 0
            self._lr_warmup_factor = 1.0


        self._learning_velocity = LearningVelocity()
        at = new_config.adaptive_thresholds
        self._learning_velocity.plateau_threshold = float(at.plateau_improvement_threshold)


        if reason == "demotion" and demoted_from_stage is not None:
            recovery_config = new_config.recovery_protocol


            self._recovery_state = self._demotion_analyzer.create_recovery_protocol(
                demoted_from_stage, recovery_config
            )
            if self._recovery_state.is_active() and self.verbose:
                logger.info(
                    f"  Recovery protocol activated: {self._recovery_state.trigger_reason}, "
                    f"{self._recovery_state.episodes_remaining} episodes"
                )
        else:

            self._recovery_state = RecoveryProtocolState()

        if self.verbose:
            logger.info(f"Entered stage {stage.name} (epoch={self._current_stage_epoch}, reason={reason})")
            if self._lr_warmup_active:
                logger.info(f"  LR warmup: {transition.lr_warmup_factor:.0%} → 100% over {transition.lr_warmup_steps:,} steps")
            if self._reward_blend_remaining > 0:
                logger.info(f"  Reward blend: {self._reward_blend_remaining} episodes")


    @property
    def stage_config(self) -> CurriculumStageConfig:
        return get_stage_config(self.current_stage)

    @property
    def competence_thresholds(self) -> CompetenceThresholds:
        base = self.stage_config.competence
        config = self.stage_config.adaptive_thresholds
        promotion_threshold = self.stage_config.composite_scoring.promotion_threshold
        return compute_adjusted_thresholds(
            base,
            self._learning_velocity,
            config,
            composite_score=self._composite_score,
            promotion_threshold=promotion_threshold,
        )

    @property
    def current_stage_epoch(self) -> int:
        return self._current_stage_epoch

    @property
    def is_in_transition(self) -> bool:
        return self._lr_warmup_active or self._reward_blend_remaining > 0

    def is_learning_plateaued(self) -> bool:
        threshold = self.stage_config.adaptive_thresholds.plateau_episodes_threshold
        return self._learning_velocity.is_plateaued(threshold)

    @property
    def reward_blend_factor(self) -> float:
        if self._reward_blend_remaining <= 0 or self._previous_stage_config is None:
            return 1.0

        transition = self.stage_config.transition
        total_blend = transition.reward_blend_episodes
        if total_blend <= 0:
            return 1.0

        progress = 1.0 - (self._reward_blend_remaining / total_blend)
        return _clamp(progress, 0.0, 1.0)

    @property
    def learning_velocity(self) -> LearningVelocity:
        return self._learning_velocity

    @property
    def skill_assessment(self) -> Optional[SkillAssessment]:
        return self._skill_assessment

    @property
    def recovery_state(self) -> RecoveryProtocolState:
        return self._recovery_state

    @property
    def review_state(self) -> ReviewSessionState:
        return self._review_state

    @property
    def composite_score(self) -> Optional[CompositeScore]:
        return self._composite_score


    def get_lr_multiplier(self, base_steps_since_transition: int = 0) -> float:
        if not self._lr_warmup_active:
            return 1.0

        transition = self.stage_config.transition
        if transition.lr_warmup_steps <= 0:
            return 1.0

        warmup_progress = 1.0 - (self._lr_warmup_steps_remaining / transition.lr_warmup_steps)
        warmup_progress = _clamp(warmup_progress, 0.0, 1.0)

        return self._lr_warmup_factor + (1.0 - self._lr_warmup_factor) * warmup_progress

    def on_train_step(self, timesteps: int = 1) -> None:
        if self._lr_warmup_active and self._lr_warmup_steps_remaining > 0:
            self._lr_warmup_steps_remaining -= timesteps
            if self._lr_warmup_steps_remaining <= 0:
                self._lr_warmup_active = False
                self._lr_warmup_factor = 1.0
                if self.verbose:
                    logger.info(f"LR warmup complete for stage {self.current_stage.name}")


    def step_transition_state(self, timesteps: int = 1) -> None:
        self.on_train_step(timesteps)

    def get_effective_stage_config(self) -> CurriculumStageConfig:
        # get_effective_stage() is episode-cached.  Calling it here makes mixed
        # stage rehearsal and review sessions affect the actual configuration
        # consumed by the environment while guaranteeing that reset() and this
        # method see the same sampled stage for the episode.
        effective_stage = self.get_effective_stage()
        cfg = copy.deepcopy(get_stage_config(effective_stage))


        if self._reward_blend_remaining > 0 and self._previous_stage_config is not None:
            alpha = self.reward_blend_factor
            prev_rewards = self._previous_stage_config.rewards
            curr_rewards = cfg.rewards
            for field_obj in fields(curr_rewards):
                field_name = field_obj.name
                if not hasattr(prev_rewards, field_name):
                    continue
                prev_val = getattr(prev_rewards, field_name, None)
                curr_val = getattr(curr_rewards, field_name, None)
                if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)):
                    blended_val = prev_val * (1 - alpha) + curr_val * alpha
                    setattr(cfg.rewards, field_name, blended_val)


        if self._recovery_state.is_active():

            for k, v in (self._recovery_state.reward_modifications or {}).items():
                if hasattr(cfg.rewards, k) and isinstance(v, (int, float)):
                    cur = getattr(cfg.rewards, k)
                    if isinstance(cur, (int, float)):
                        setattr(cfg.rewards, k, cur * float(v))
                    else:
                        setattr(cfg.rewards, k, v)


            for k, v in (self._recovery_state.constraint_modifications or {}).items():
                if hasattr(cfg.constraints, k) and isinstance(v, (int, float)):
                    cur = getattr(cfg.constraints, k)
                    if isinstance(cur, bool):
                        setattr(cfg.constraints, k, bool(v))
                    elif isinstance(cur, int):
                        setattr(cfg.constraints, k, max(1, int(round(cur * float(v)))))
                    elif isinstance(cur, float):
                        setattr(cfg.constraints, k, cur * float(v))
                    else:
                        setattr(cfg.constraints, k, v)


        if self._selectivity_phase_active:
            req = self._selectivity_phase_requirements or {}
            max_trades = int(req.get("max_trades_per_episode", 0) or 0)
            if max_trades > 0:
                cfg.constraints.max_trades_per_episode = max_trades
            min_setup = float(req.get("min_setup_quality", 0.0) or 0.0)
            if min_setup > 0:
                cfg.constraints.min_setup_quality_for_entry = min_setup
                if hasattr(cfg.rewards, "setup_quality_threshold"):
                    cfg.rewards.setup_quality_threshold = max(cfg.rewards.setup_quality_threshold, min_setup)
            min_certainty = float(req.get("min_entry_certainty", 0.0) or 0.0)
            if min_certainty > 0:
                cfg.rewards.certainty_threshold = max(cfg.rewards.certainty_threshold, min_certainty)

        # CurriculumStageConfig computes override dictionaries only once in
        # __post_init__.  Reward blending and recovery mutate the dataclass
        # afterwards, so without rebuilding this channel the environment sees
        # the original static values.  Materialise every RewardShaping field,
        # including fields added after the original allow-list was written.
        cfg.reward_overrides = {
            field_obj.name: copy.deepcopy(getattr(cfg.rewards, field_obj.name))
            for field_obj in fields(cfg.rewards)
        }

        # Recovery also mutates constraints.  Rebuild this channel so its
        # original static values cannot overwrite those effective constraints
        # when the environment applies the dictionaries after direct mappings.
        cfg.env_overrides = cfg._compute_env_overrides()

        return cfg


    def start_selectivity_phase(
        self,
        episodes: int = 20,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> None:
        if episodes <= 0:
            return

        default_requirements = {
            "max_trades_per_episode": 3,
            "min_setup_quality": 0.75,
            "min_entry_certainty": 0.70,
            "required_success_rate": 0.67,
        }
        req = {**default_requirements, **(requirements or {})}

        self._selectivity_phase_active = True
        self._selectivity_phase_target_episodes = int(episodes)
        self._selectivity_phase_episodes_remaining = int(episodes)
        self._selectivity_phase_failures = 0
        self._selectivity_phase_requirements = req

        if self.verbose:
            logger.info(
                f"⚡ Selectivity Mastery Phase started: {episodes} episodes, "
                f"max_trades_per_episode={req.get('max_trades_per_episode')}, "
                f"min_setup_quality={req.get('min_setup_quality')}"
            )

    def check_selectivity_phase(self, metrics: EpisodeMetrics) -> Dict[str, Any]:
        if not self._selectivity_phase_active:
            return {"active": False}

        req = self._selectivity_phase_requirements or {}
        result: Dict[str, Any] = {
            "active": True,
            "episodes_remaining": self._selectivity_phase_episodes_remaining,
            "passed": True,
            "violations": [],
        }

        max_trades = int(req.get("max_trades_per_episode", 0) or 0)
        if max_trades > 0 and int(metrics.trade_count) > max_trades:
            result["passed"] = False
            result["violations"].append(
                f"Too many trades: {metrics.trade_count} > {max_trades}"
            )

        min_setup_quality = float(req.get("min_setup_quality", 0.0) or 0.0)
        if min_setup_quality > 0 and float(metrics.avg_setup_quality) < min_setup_quality:
            result["passed"] = False
            result["violations"].append(
                f"Setup quality too low: {metrics.avg_setup_quality:.2f} < {min_setup_quality}"
            )

        min_certainty = float(req.get("min_entry_certainty", 0.0) or 0.0)
        if min_certainty > 0 and float(metrics.avg_entry_certainty) < min_certainty:
            result["passed"] = False
            result["violations"].append(
                f"Entry certainty too low: {metrics.avg_entry_certainty:.2f} < {min_certainty}"
            )

        # Quality/certainty averages without a trade are not evidence.  A flat
        # episode is still allowed, but it cannot be counted as a successful
        # selectivity demonstration merely by carrying synthetic averages.
        if int(metrics.trade_count) <= 0 and (min_setup_quality > 0 or min_certainty > 0):
            result["passed"] = False
            result["violations"].append(
                "No executed trade to evidence setup quality and entry certainty"
            )

        if not result["passed"]:
            self._selectivity_phase_failures += 1

        self._selectivity_phase_episodes_remaining -= 1
        target_episodes = max(1, int(self._selectivity_phase_target_episodes))
        episodes_remaining = max(0, int(self._selectivity_phase_episodes_remaining))
        episodes_evaluated = min(target_episodes, target_episodes - episodes_remaining)
        failed_episodes = min(episodes_evaluated, int(self._selectivity_phase_failures))
        successful_episodes = max(0, episodes_evaluated - failed_episodes)
        success_rate = successful_episodes / episodes_evaluated if episodes_evaluated > 0 else 0.0
        required_success_rate = _clamp(
            _safe_float(req.get("required_success_rate", 0.0), 0.0),
            0.0,
            1.0,
        )

        result.update({
            "episodes_remaining": episodes_remaining,
            "episodes_evaluated": episodes_evaluated,
            "minimum_evidence_episodes": target_episodes,
            "successful_episodes": successful_episodes,
            "failed_episodes": failed_episodes,
            "success_rate": success_rate,
            "required_success_rate": required_success_rate,
        })

        if self._selectivity_phase_episodes_remaining <= 0:
            evidence_complete = episodes_evaluated >= target_episodes
            phase_passed = evidence_complete and success_rate >= required_success_rate
            result["evidence_complete"] = evidence_complete
            result["phase_passed"] = phase_passed

            if phase_passed:
                self._selectivity_phase_active = False
                self._selectivity_phase_completed_epoch = self._current_stage_epoch
                result["phase_complete"] = True
                if self.verbose:
                    logger.info(
                        "✅ Selectivity Mastery Phase completed: "
                        f"{successful_episodes}/{episodes_evaluated} episodes "
                        f"({success_rate:.1%} >= {required_success_rate:.1%})"
                    )
            else:
                self._selectivity_phase_episodes_remaining = self._selectivity_phase_target_episodes
                self._selectivity_phase_failures = 0
                result["phase_reset"] = True
                if self.verbose:
                    logger.info(
                        "🔁 Selectivity phase reset: "
                        f"{successful_episodes}/{episodes_evaluated} episodes "
                        f"({success_rate:.1%} < {required_success_rate:.1%})"
                    )

        return result

    def _seen_episode_id(self, eid: int) -> bool:
        return eid in self._processed_episode_id_set

    def _mark_episode_id(self, eid: int) -> None:

        if len(self._processed_episode_ids) >= PROCESSED_EPISODE_ID_LIMIT:
            old = self._processed_episode_ids.popleft()
            self._processed_episode_id_set.discard(old)
        self._processed_episode_ids.append(eid)
        self._processed_episode_id_set.add(eid)

    def episode_transition_tick(self) -> None:
        if self._transition_cooldown_remaining > 0:
            self._transition_cooldown_remaining -= 1

        if self._reward_blend_remaining > 0:
            self._reward_blend_remaining -= 1
            if self._reward_blend_remaining <= 0 and self.verbose:
                logger.info(f"Reward blending complete for stage {self.current_stage.name}")


        if self._recovery_state.is_active():
            self._recovery_state.tick()
            if not self._recovery_state.is_active() and self.verbose:
                logger.info(f"Recovery protocol complete for stage {self.current_stage.name}")


    def on_episode_end(
        self,
        metrics: EpisodeMetrics,
        timesteps: int = 0,
        effective_stage: Optional[CurriculumStage] = None,
        check_transitions: bool = True,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "episode_idx": self.total_episodes + 1,
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "promoted": False,
            "demoted": False,
            "transition_to": None,
            "invariant_violations": [],
        }


        gid = int(getattr(metrics, "global_episode_idx", 0) or 0)


        if gid <= 0 or gid <= self.total_episodes:
            episode_id = self.total_episodes + 1
        else:
            episode_id = gid

        if self._seen_episode_id(episode_id):
            logger.debug(f"on_episode_end: duplicate episode_id={episode_id}, skipping")
            result["skipped"] = True
            result["skip_reason"] = f"duplicate_episode_id:{episode_id}"
            return result


        next_idx = self.total_episodes + 1
        if next_idx <= self._last_episode_end_idx:
            logger.warning(
                f"on_episode_end called for episode {next_idx} but already processed "
                f"up to {self._last_episode_end_idx}. Skipping to maintain counter integrity."
            )
            result["skipped"] = True
            return result


        violations = self._invariant_checker.check_episode(
            metrics=asdict(metrics),
            stage_name=self.current_stage.name,
            stage_epoch=self._current_stage_epoch,
            episode_idx=next_idx,
        )
        result["invariant_violations"] = [str(v) for v in violations]


        self.record_episode(metrics, timesteps=timesteps, effective_stage=effective_stage)


        selectivity_result = self.check_selectivity_phase(metrics)
        if selectivity_result.get("active", False):
            result["selectivity_phase"] = selectivity_result


        self._last_episode_end_idx = self.total_episodes
        self._mark_episode_id(episode_id)


        if check_transitions and self._transition_cooldown_remaining <= 0:

            if self.auto_promote:
                promoted, new_stage = self.try_promote()
                result["promoted"] = promoted
                if promoted:
                    result["transition_to"] = new_stage.name if new_stage else None


            if not result["promoted"] and self.auto_demote:
                demoted, new_stage = self.try_demote()
                result["demoted"] = demoted
                if demoted:
                    result["transition_to"] = new_stage.name if new_stage else None

        result["stage_after"] = self.current_stage.name
        result["cooldown_remaining"] = self._transition_cooldown_remaining

        return result


    def update_entropy(self, entropy: float) -> None:
        self._current_entropy = entropy

    def get_entropy_penalty(self) -> float:
        if self._current_entropy < 0:
            return 0.0

        targets = self.stage_config.entropy_targets

        if self._current_entropy < targets.min_entropy:
            return (targets.min_entropy - self._current_entropy) * targets.low_entropy_penalty_scale
        elif self._current_entropy > targets.max_entropy:
            return (self._current_entropy - targets.max_entropy) * targets.high_entropy_penalty_scale

        return 0.0


    def sample_training_stage(self) -> CurriculumStage:
        config = self.stage_config.mixed_stage_sampling

        if not config.enabled:
            return self.current_stage

        r = self._rng.random()


        total_weight = (
            config.current_stage_weight +
            config.recent_stages_weight +
            config.foundation_weight
        )
        if total_weight <= 0:
            return self.current_stage

        current_prob = config.current_stage_weight / total_weight
        recent_prob = config.recent_stages_weight / total_weight


        if r < current_prob:
            return self.current_stage
        elif r < current_prob + recent_prob:

            cur_idx = _stage_to_index(self.current_stage)
            min_idx = max(0, cur_idx - config.recent_stage_depth)
            if min_idx >= cur_idx:
                return self.current_stage
            sampled_idx = int(self._rng.integers(min_idx, cur_idx))
            return _index_to_stage(sampled_idx)
        else:

            return _foundation_stage()


    def check_review_session(self) -> Optional[CurriculumStage]:
        config = self.stage_config.review_session

        if not config.enabled:
            return None


        if config.min_stage_for_review is None:
            return None


        cur_idx = _stage_to_index(self.current_stage)
        min_review_idx = _stage_to_index(config.min_stage_for_review)
        if cur_idx < min_review_idx:
            return None


        next_ep = self.total_episodes + 1
        already_ticked = (self._review_tick_episode == next_ep)
        if not already_ticked:
            self._review_tick_episode = next_ep


        if self._review_state.in_review:

            if not already_ticked:
                self._review_state.review_episodes_remaining -= 1
            if self._review_state.review_episodes_remaining <= 0:


                final_review_stage = self._review_state.review_stage


                self._review_state.in_review = False
                self._review_state.episodes_since_review = 0
                self._review_state.return_to_home_latch = True
                if self.verbose:
                    home_name = self._review_state.home_stage.name if self._review_state.home_stage else "current"
                    logger.info(f"Review session complete, returning to {home_name} next episode")
                return final_review_stage
            return self._review_state.review_stage


        if not already_ticked:
            self._review_state.episodes_since_review += 1

        if self._review_state.episodes_since_review >= config.review_frequency:

            cur_idx = _stage_to_index(self.current_stage)
            min_review_idx = max(0, cur_idx - config.review_depth)
            if min_review_idx < cur_idx:
                review_stage_idx = int(self._rng.integers(min_review_idx, cur_idx))
                review_stage = _index_to_stage(review_stage_idx)

                self._review_state.in_review = True
                self._review_state.review_stage = review_stage
                self._review_state.review_episodes_remaining = config.review_duration
                self._review_state.home_stage = self.current_stage

                if self.verbose:
                    logger.info(f"Starting review session on {review_stage.name} ({config.review_duration} episodes)")

                return review_stage

        return None

    def get_effective_stage(self) -> CurriculumStage:

        next_ep = self.total_episodes + 1
        if (self._cached_effective_stage_episode == next_ep
            and self._cached_effective_stage is not None):
            return self._cached_effective_stage


        stage: CurriculumStage


        if getattr(self._review_state, 'return_to_home_latch', False):
            self._review_state.return_to_home_latch = False
            stage = self._review_state.home_stage or self.current_stage
        else:

            review_stage = self.check_review_session()
            if review_stage is not None:
                stage = review_stage
            else:

                stage = self.sample_training_stage()


        self._cached_effective_stage = stage
        self._cached_effective_stage_episode = next_ep

        return stage

    def _clear_effective_stage_cache(self) -> None:
        self._cached_effective_stage = None
        self._cached_effective_stage_episode = -1


    def record_episode(
        self,
        metrics: EpisodeMetrics,
        timesteps: int = 0,
        effective_stage: Optional[CurriculumStage] = None,
    ) -> None:


        record_to_stage = effective_stage if effective_stage is not None else self.current_stage


        if effective_stage is not None:
            metrics.stage_name = record_to_stage.name
        else:
            metrics.stage_name = metrics.stage_name or record_to_stage.name


        if record_to_stage == self.current_stage:
            metrics.stage_epoch = metrics.stage_epoch or self._current_stage_epoch
        else:
            metrics.stage_epoch = self._stage_epoch_counter.get(record_to_stage, 0)

        metrics.global_episode_idx = metrics.global_episode_idx or (self.total_episodes + 1)
        metrics.policy_entropy = self._current_entropy
        if not metrics.timestamp:
            metrics.timestamp = _now_iso(self.tz)


        self._history[record_to_stage].append(metrics)


        self._stage_timesteps_total[record_to_stage] += timesteps
        self._stage_episodes_total[record_to_stage] += 1


        if record_to_stage == self.current_stage:
            self.stage_timesteps += timesteps
            self.stage_episodes += 1

        self.total_timesteps += timesteps
        self.total_episodes += 1
        self._rolling_stats_dirty = True


        self._clear_effective_stage_cache()


        if record_to_stage == self.current_stage:
            self._learning_velocity.update({
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "avg_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown,
            })


        self.episode_transition_tick()

    def record_episode_from_info(
        self,
        info: Dict[str, Any],
        episode_reward: float,
        episode_length: int,
        effective_stage: Optional[CurriculumStage] = None,
    ) -> None:
        ep_stats = info.get("episode_stats", {}) or {}


        total_pnl = _safe_float(info.get("total_pnl", ep_stats.get("total_pnl", 0.0)), 0.0)
        win_rate = _clamp(_safe_float(info.get("win_rate", ep_stats.get("win_rate", 0.0)), 0.0), 0.0, 1.0)


        trade_count = _safe_int(info.get("trade_count", ep_stats.get("trade_count", 0)), 0)
        winning_trades = _safe_int(ep_stats.get("winning_trades", info.get("winning_trades", 0)), 0)
        losing_trades = _safe_int(ep_stats.get("losing_trades", info.get("losing_trades", 0)), 0)


        computed_sum = winning_trades + losing_trades
        if computed_sum > trade_count and trade_count > 0:

            logger.warning(
                f"Trade accounting mismatch: wins({winning_trades})+losses({losing_trades})={computed_sum} > "
                f"trade_count({trade_count}). Using wins+losses as denominator."
            )
            trade_count = computed_sum
        elif (winning_trades == 0 and losing_trades == 0) and trade_count > 0:

            approx_wins = int(round(win_rate * trade_count))
            winning_trades = max(0, min(trade_count, approx_wins))
            losing_trades = max(0, trade_count - winning_trades)


        max_dd = _clamp(_safe_float(info.get("max_drawdown", ep_stats.get("max_drawdown", info.get("drawdown", 0.0))), 0.0), 0.0, 1.0)
        daily_dd = _clamp(_safe_float(info.get("daily_drawdown", ep_stats.get("daily_drawdown", 0.0)), 0.0), 0.0, 1.0)

        termination_reason = str(info.get("termination_reason", ep_stats.get("termination_reason", "")) or "")

        dd_breach = bool(info.get("dd_breach", ep_stats.get("dd_breach", False)))
        if not dd_breach and termination_reason:
            low = termination_reason.lower()
            dd_breach = ("drawdown" in low) or ("dd breach" in low) or ("daily_dd" in low) or ("max_dd" in low)

        pf_raw = ep_stats.get("profit_factor", info.get("profit_factor", 0.0))
        pf = _safe_float(pf_raw, 0.0)
        if math.isinf(pf) or pf > 1e6:
            pf = 10.0
        pf = _clamp(pf, 0.0, 10.0)

        hit_max_consec = bool(info.get("hit_max_consecutive_losses", ep_stats.get("hit_max_consecutive_losses", False)))
        if not hit_max_consec and termination_reason:
            hit_max_consec = "consecutive" in termination_reason.lower()


        exit_dist = ep_stats.get("exit_quality_distribution", {}) or {}


        GOOD_EXITS = {"trailing_stop", "agent_close"}
        BAD_EXITS = {"hard_stop", "emergency_close", "risk_liquidation"}
        NEUTRAL_EXITS = {"time_decay", "hard_close", "weekend_flatten",
                        "daily_limit_safety", "episode_truncate_flatten", "episode_truncate"}

        trailing_stops = _safe_int(exit_dist.get("trailing_stop", 0), 0)
        agent_closes = _safe_int(exit_dist.get("agent_close", 0), 0)


        hard_stops = _safe_int(exit_dist.get("hard_stop", 0), 0) + _safe_int(exit_dist.get("emergency_close", 0), 0)

        risk_liquidations = _safe_int(exit_dist.get("risk_liquidation", 0), 0)


        neutral_exits = sum(
            _safe_int(exit_dist.get(key, 0), 0)
            for key in NEUTRAL_EXITS
        )


        known_keys = GOOD_EXITS | BAD_EXITS | NEUTRAL_EXITS
        unknown_exits = sum(
            _safe_int(count, 0)
            for key, count in exit_dist.items()
            if key not in known_keys
        )
        if unknown_exits > 0:
            logger.debug(f"Found {unknown_exits} exits with unrecognized types, treating as neutral")

        tracked_exits = trailing_stops + agent_closes + hard_stops + risk_liquidations + neutral_exits + unknown_exits
        raw_other_exits = trade_count - tracked_exits


        if raw_other_exits < 0:
            logger.debug(f"Exit count mismatch: trade_count={trade_count}, tracked={tracked_exits}")
            raw_other_exits = 0

        other_exits = raw_other_exits + neutral_exits + unknown_exits

        metrics = EpisodeMetrics(
            total_pnl=total_pnl,
            win_rate=win_rate,
            trade_count=trade_count,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            max_drawdown=max_dd,
            daily_drawdown=daily_dd,
            dd_breach=dd_breach,
            avg_r_multiple=_safe_float(ep_stats.get("avg_r_multiple", info.get("avg_r_multiple", 0.0)), 0.0),
            profit_factor=pf,
            avg_mae=_safe_float(ep_stats.get("avg_mae", info.get("avg_mae", 0.0)), 0.0),
            avg_mfe=_safe_float(ep_stats.get("avg_mfe", info.get("avg_mfe", 0.0)), 0.0),
            avg_bars_held=_safe_float(ep_stats.get("avg_bars_held", info.get("avg_bars_held", 0.0)), 0.0),
            avg_entry_quality=_clamp(_safe_float(ep_stats.get("avg_entry_quality", info.get("avg_entry_quality", 0.5)), 0.5), 0.0, 1.0),
            avg_bars_between_trades=_safe_float(ep_stats.get("avg_bars_between_trades", info.get("avg_bars_between_trades", 0.0)), 0.0),
            avg_setup_quality=_clamp(_safe_float(ep_stats.get("avg_setup_quality", info.get("avg_setup_quality", 0.0)), 0.0), 0.0, 1.0),
            avg_entry_certainty=_clamp(_safe_float(ep_stats.get("avg_entry_certainty", info.get("avg_entry_certainty", 0.0)), 0.0), 0.0, 1.0),
            min_setup_quality_for_entry=_clamp(_safe_float(ep_stats.get("min_setup_quality_for_entry", info.get("min_setup_quality_for_entry", 0.0)), 0.0), 0.0, 1.0),
            consecutive_losses=_safe_int(info.get("consecutive_losses", ep_stats.get("consecutive_losses", 0)), 0),
            consecutive_wins=_safe_int(info.get("consecutive_wins", ep_stats.get("consecutive_wins", 0)), 0),
            max_consecutive_losses_reached=_safe_int(ep_stats.get("max_consecutive_losses_reached",
                                                                   info.get("consecutive_losses", ep_stats.get("consecutive_losses", 0))), 0),
            hit_max_consecutive_losses=hit_max_consec,
            mask_collapse_steps=_safe_int(ep_stats.get("mask_collapse_steps", info.get("mask_collapse_steps", 0)), 0),
            mask_decision_steps=_safe_int(ep_stats.get("mask_decision_steps", info.get("mask_decision_steps", 0)), 0),
            stop_mode_steps=_safe_int(ep_stats.get("stop_mode_steps", info.get("stop_mode_steps", 0)), 0),
            setup_skipped_count=_safe_int(ep_stats.get("setup_skipped_count", info.get("setup_skipped_count", 0)), 0),
            fomo_trade_count=_safe_int(ep_stats.get("fomo_trade_count", info.get("fomo_trade_count", 0)), 0),
            revenge_trade_count=_safe_int(ep_stats.get("revenge_trade_count", info.get("revenge_trade_count", 0)), 0),
            max_patience_bars=_safe_int(ep_stats.get("max_patience_bars", info.get("max_patience_bars", 0)), 0),
            trailing_stop_exits=trailing_stops,
            agent_close_exits=agent_closes,
            hard_stop_exits=hard_stops,
            risk_liquidation_exits=risk_liquidations,
            other_exits=other_exits,
            episode_length=_safe_int(episode_length, 0),
            episode_reward=_safe_float(episode_reward, 0.0),
            termination_reason=termination_reason,
            stage_name=self.current_stage.name,
            stage_epoch=self._current_stage_epoch,
            global_episode_idx=self.total_episodes + 1,
            policy_entropy=self._current_entropy,
            timestamp=_now_iso(self.tz),
        )


        self._accumulate_regime_trades(ep_stats.get("trades_with_regime", []))


        self.on_episode_end(
            metrics=metrics,
            timesteps=episode_length,
            effective_stage=effective_stage,
            check_transitions=True,
        )

    def _accumulate_regime_trades(self, trades_data: List[Dict[str, Any]]) -> None:
        if not trades_data:
            return


        from envs.curriculum.regime_skill_assessment import SessionRegime, SpreadRegime, TrendRegime, VolatilityRegime

        vol_map = {
            "low": VolatilityRegime.LOW,
            "medium": VolatilityRegime.MEDIUM,
            "high": VolatilityRegime.HIGH,
        }
        trend_map = {
            "strong_trend": TrendRegime.STRONG_TREND,
            "weak_trend": TrendRegime.WEAK_TREND,
            "ranging": TrendRegime.RANGING,
        }
        session_map = {
            "asian": SessionRegime.ASIAN,
            "london": SessionRegime.LONDON,
            "overlap": SessionRegime.LONDON_NY_OVERLAP,
            "ny": SessionRegime.NY,
            "off_hours": SessionRegime.OFF_HOURS,
        }
        spread_map = {
            "tight": SpreadRegime.TIGHT,
            "normal": SpreadRegime.NORMAL,
            "wide": SpreadRegime.WIDE,
        }

        for td in trades_data:
            try:
                trade = TradeWithRegime(
                    pnl=float(td.get("pnl", 0.0)),
                    r_multiple=float(td.get("r_multiple", 0.0)),
                    is_winner=bool(td.get("is_winner", False)),
                    bars_held=int(td.get("bars_held", 0)),
                    mae=float(td.get("mae", 0.0)),
                    mfe=float(td.get("mfe", 0.0)),
                    entry_quality=float(td.get("entry_quality", 0.5)),
                    exit_type=str(td.get("exit_type", "")),
                    volatility_regime=vol_map.get(td.get("volatility_regime", "medium"), VolatilityRegime.MEDIUM),
                    trend_regime=trend_map.get(td.get("trend_regime", "ranging"), TrendRegime.RANGING),
                    session_regime=session_map.get(td.get("session_regime", "off_hours"), SessionRegime.OFF_HOURS),
                    spread_regime=spread_map.get(td.get("spread_regime", "normal"), SpreadRegime.NORMAL),
                )
                self._regime_trades.append(trade)
                self._regime_assessment_dirty = True
            except Exception as e:
                logger.debug(f"Failed to parse regime trade: {e}")

    def _update_regime_assessment(self) -> None:
        if not self._regime_assessment_dirty:
            return

        if len(self._regime_trades) >= 50:
            self._regime_assessment = RegimeSkillAssessment.from_trades(list(self._regime_trades))
            self._regime_assessment_dirty = False
        else:
            self._regime_assessment = None


    def _current_epoch_window(self, window_size: int) -> List[EpisodeMetrics]:
        history = list(self._history[self.current_stage])
        epoch = self._current_stage_epoch
        filtered = [m for m in history if m.stage_epoch == epoch]
        return filtered[-window_size:] if len(filtered) > window_size else filtered

    def get_rolling_stats(self, force_refresh: bool = False) -> RollingStats:
        if not force_refresh and not self._rolling_stats_dirty and self._rolling_stats is not None:
            return self._rolling_stats

        thresholds = self.stage_config.competence
        window_size = thresholds.evaluation_window

        window = self._current_epoch_window(window_size)

        if not window:
            self._rolling_stats = RollingStats(0)
            self._rolling_stats_dirty = False
            return self._rolling_stats


        pnls = np.array([_safe_float(m.total_pnl, 0.0) for m in window], dtype=np.float64)
        win_rates = np.array([_clamp(_safe_float(m.win_rate, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        trade_counts = np.array([max(0.0, float(m.trade_count)) for m in window], dtype=np.float64)
        episode_lengths = np.array([max(0.0, float(getattr(m, "episode_length", 0))) for m in window], dtype=np.float64)
        mask_collapse_steps = np.array([max(0.0, float(getattr(m, "mask_collapse_steps", 0))) for m in window], dtype=np.float64)
        stop_mode_steps = np.array([max(0.0, float(getattr(m, "stop_mode_steps", 0))) for m in window], dtype=np.float64)
        drawdowns = np.array([_clamp(_safe_float(m.max_drawdown, 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        dd_breaches = np.array([bool(m.dd_breach) for m in window], dtype=np.bool_)
        profit_factors = np.array([_clamp(_safe_float(m.profit_factor, 0.0), 0.0, 10.0) for m in window], dtype=np.float64)
        r_multiples = np.array([_safe_float(m.avg_r_multiple, 0.0) for m in window], dtype=np.float64)
        entry_qualities = np.array([_clamp(_safe_float(m.avg_entry_quality, 0.5), 0.0, 1.0) for m in window], dtype=np.float64)
        bars_between = np.array([max(0.0, _safe_float(getattr(m, "avg_bars_between_trades", 0.0), 0.0)) for m in window], dtype=np.float64)
        setup_skipped = np.array([max(0.0, _safe_float(getattr(m, "setup_skipped_count", 0.0), 0.0)) for m in window], dtype=np.float64)
        entry_certainty = np.array([_clamp(_safe_float(getattr(m, "avg_entry_certainty", 0.0), 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        setup_quality = np.array([_clamp(_safe_float(getattr(m, "avg_setup_quality", 0.0), 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        min_setup_quality = np.array([_clamp(_safe_float(getattr(m, "min_setup_quality_for_entry", 0.0), 0.0), 0.0, 1.0) for m in window], dtype=np.float64)
        fomo_counts = np.array([max(0.0, _safe_float(getattr(m, "fomo_trade_count", 0.0), 0.0)) for m in window], dtype=np.float64)
        revenge_counts = np.array([max(0.0, _safe_float(getattr(m, "revenge_trade_count", 0.0), 0.0)) for m in window], dtype=np.float64)
        max_patience_bars = np.array([max(0.0, _safe_float(getattr(m, "max_patience_bars", 0.0), 0.0)) for m in window], dtype=np.float64)
        consec_loss_breaches = np.array([bool(m.hit_max_consecutive_losses) for m in window], dtype=np.bool_)

        max_consec_losses = np.array([max(0, m.max_consecutive_losses_reached) for m in window], dtype=np.int32)
        entropies = np.array([m.policy_entropy for m in window if m.policy_entropy >= 0], dtype=np.float64)


        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        mean_win_rate = float(np.mean(win_rates))
        std_win_rate = float(np.std(win_rates, ddof=1)) if len(win_rates) > 1 else 0.0
        mean_trade_count = float(np.mean(trade_counts))

        total_trades = int(np.sum(trade_counts))
        total_wins = int(sum(max(0, m.winning_trades) for m in window))
        total_losses = int(sum(max(0, m.losing_trades) for m in window))
        fomo_trade_rate = float(np.sum(fomo_counts) / max(total_trades, 1))
        revenge_trade_rate = float(np.sum(revenge_counts) / max(total_trades, 1))


        min_trades_per_ep = int(getattr(thresholds, "min_trades_per_episode_for_win_rate_stability", 0) or 0)
        if min_trades_per_ep < 0:
            min_trades_per_ep = 0

        eligible_mask = (trade_counts >= float(min_trades_per_ep)) if min_trades_per_ep > 0 else (trade_counts > 0)
        eligible_win_rates = win_rates[eligible_mask]
        eligible_trade_counts = trade_counts[eligible_mask]
        eligible_episodes = int(np.sum(eligible_mask))
        eligible_trades = int(np.sum(eligible_trade_counts)) if eligible_episodes > 0 else 0

        std_win_rate_trade_weighted = 0.0
        if eligible_episodes >= 2 and eligible_trades > 0:
            wsum = float(np.sum(eligible_trade_counts))
            wmean = float(np.sum(eligible_trade_counts * eligible_win_rates) / max(wsum, 1e-12))
            wvar = float(np.sum(eligible_trade_counts * (eligible_win_rates - wmean) ** 2) / max(wsum, 1e-12))
            std_win_rate_trade_weighted = float(np.sqrt(max(wvar, 0.0)))

        std_win_rate_eligible_unweighted = (
            float(np.std(eligible_win_rates, ddof=1)) if eligible_episodes > 1 else 0.0
        )

        mean_drawdown = float(np.mean(drawdowns))
        max_drawdown_seen = float(np.max(drawdowns)) if len(drawdowns) else 0.0
        dd_breach_rate = float(np.mean(dd_breaches)) if len(dd_breaches) else 0.0


        mean_profit_factor = float(np.mean(profit_factors)) if len(profit_factors) else 0.0

        mean_r_multiple = float(np.mean(r_multiples)) if len(r_multiples) else 0.0
        mean_entry_quality = float(np.mean(entry_qualities)) if len(entry_qualities) else 0.5
        mean_bars_between_trades = float(np.mean(bars_between)) if len(bars_between) else 0.0
        std_bars_between_trades = float(np.std(bars_between, ddof=1)) if len(bars_between) > 1 else 0.0
        mean_setup_skipped_per_episode = float(np.mean(setup_skipped)) if len(setup_skipped) else 0.0
        mean_entry_certainty = float(np.mean(entry_certainty)) if len(entry_certainty) else 0.0
        mean_setup_quality = float(np.mean(setup_quality)) if len(setup_quality) else 0.0
        mean_min_setup_quality = float(np.mean(min_setup_quality)) if len(min_setup_quality) else 0.0
        mean_max_patience_bars = float(np.mean(max_patience_bars)) if len(max_patience_bars) else 0.0
        consecutive_loss_breach_rate = float(np.mean(consec_loss_breaches)) if len(consec_loss_breaches) else 0.0

        avg_max_consecutive_losses = float(np.mean(max_consec_losses)) if len(max_consec_losses) else 0.0

        consecutive_loss_streak_rate = float(np.mean(max_consec_losses >= 3)) if len(max_consec_losses) else 0.0


        # Denominator must be the number of mask-tracking calls, not the number
        # of env steps. _track_action_mask_state_for_metrics runs from two call
        # sites per step, so dividing its counts by episode length produced
        # rates above 1.0 - a live run reported mask_collapse_rate 1.1311
        # against a 1.0 limit and listed it as a promotion blocker that no
        # policy could clear. The env already reports the matching denominator.
        decision_steps = np.array(
            [max(0.0, float(getattr(m, "mask_decision_steps", 0))) for m in window],
            dtype=np.float64,
        )
        total_decisions = float(np.sum(decision_steps))
        if total_decisions <= 0.0:
            total_decisions = float(np.sum(episode_lengths))

        if total_decisions > 0.0:
            mask_collapse_rate = float(np.sum(mask_collapse_steps) / total_decisions)
            stop_mode_rate = float(np.sum(stop_mode_steps) / total_decisions)
        else:
            mask_collapse_rate = 0.0
            stop_mode_rate = 0.0


        if std_pnl > 1e-9:
            sharpe = mean_pnl / std_pnl
        else:
            sharpe = mean_pnl if mean_pnl > 0 else 0.0

        negative_pnls = pnls[pnls < 0]
        if negative_pnls.size:
            downside_std = float(np.std(negative_pnls))
            sortino = mean_pnl / downside_std if downside_std > 1e-9 else sharpe
        else:
            sortino = sharpe * 1.5 if mean_pnl > 0 else 0.0

        win_loss_ratio = total_wins / max(total_losses, 1)


        wl_n = max(total_trades, total_wins + total_losses, 0)
        win_low, win_high = _wilson_interval(total_wins, wl_n, z=1.96) if wl_n > 0 else (0.0, 0.0)
        win_width = float(win_high - win_low) if wl_n > 0 else 0.0


        ci_low, ci_high = _mean_ci_normal(mean_pnl, std_pnl, n=len(window), z=1.96)


        entropy_samples = len(entropies)


        mean_entropy = float(np.mean(entropies)) if len(entropies) > 0 else -1.0
        std_entropy = float(np.std(entropies)) if len(entropies) > 1 else 0.0


        total_exits = sum(
            m.trailing_stop_exits + m.agent_close_exits + m.hard_stop_exits +
            m.risk_liquidation_exits + m.other_exits
            for m in window
        )
        if total_exits > 0:
            trailing_stop_rate = sum(m.trailing_stop_exits for m in window) / total_exits
            agent_close_rate = sum(m.agent_close_exits for m in window) / total_exits
            hard_stop_rate = sum(m.hard_stop_exits for m in window) / total_exits
            risk_liquidation_rate = sum(m.risk_liquidation_exits for m in window) / total_exits
        else:
            trailing_stop_rate = agent_close_rate = hard_stop_rate = risk_liquidation_rate = 0.0

        stats = RollingStats(len(window))
        stats.mean_pnl = mean_pnl
        stats.std_pnl = std_pnl
        stats.mean_win_rate = mean_win_rate
        stats.std_win_rate = std_win_rate
        stats.std_win_rate_trade_weighted = std_win_rate_trade_weighted
        stats.std_win_rate_eligible_unweighted = std_win_rate_eligible_unweighted
        stats.win_rate_stability_eligible_episodes = eligible_episodes
        stats.win_rate_stability_eligible_trades = eligible_trades
        stats.mean_trade_count = mean_trade_count
        stats.total_trades = total_trades
        stats.total_wins = total_wins
        stats.total_losses = total_losses
        stats.win_rate_trade_weighted = float(total_wins / max(total_trades, 1))
        stats.mean_drawdown = mean_drawdown
        stats.max_drawdown_seen = max_drawdown_seen
        stats.dd_breach_rate = dd_breach_rate
        stats.mean_profit_factor = mean_profit_factor
        stats.mean_r_multiple = mean_r_multiple
        stats.mean_entry_quality = mean_entry_quality
        stats.mean_bars_between_trades = mean_bars_between_trades
        stats.std_bars_between_trades = std_bars_between_trades
        stats.mean_setup_skipped_per_episode = mean_setup_skipped_per_episode
        stats.mean_entry_certainty = mean_entry_certainty
        stats.mean_setup_quality = mean_setup_quality
        stats.mean_min_setup_quality_for_entry = mean_min_setup_quality
        stats.fomo_trade_rate = fomo_trade_rate
        stats.revenge_trade_rate = revenge_trade_rate
        stats.mean_max_patience_bars = mean_max_patience_bars
        stats.patience_consistency = std_bars_between_trades
        stats.consecutive_loss_breach_rate = consecutive_loss_breach_rate
        stats.avg_max_consecutive_losses = avg_max_consecutive_losses
        stats.consecutive_loss_streak_rate = consecutive_loss_streak_rate
        stats.mask_collapse_rate = mask_collapse_rate
        stats.stop_mode_rate = stop_mode_rate
        stats.sharpe_ratio = float(sharpe)
        stats.sortino_ratio = float(sortino)
        stats.win_loss_ratio = float(win_loss_ratio)
        stats.win_rate_wilson_low = float(win_low)
        stats.win_rate_wilson_high = float(win_high)
        stats.win_rate_wilson_width = float(win_width)
        stats.pnl_mean_ci_low = float(ci_low)
        stats.pnl_mean_ci_high = float(ci_high)
        stats.mean_entropy = mean_entropy
        stats.std_entropy = std_entropy
        stats.entropy_samples = entropy_samples
        stats.trailing_stop_rate = trailing_stop_rate
        stats.agent_close_rate = agent_close_rate
        stats.hard_stop_rate = hard_stop_rate
        stats.risk_liquidation_rate = risk_liquidation_rate

        self._rolling_stats = stats
        self._rolling_stats_dirty = False


        self._skill_assessment = SkillAssessment.from_episode_results(
            window, stats, bars_per_trading_day=self.bars_per_trading_day
        )


        self._composite_score = compute_composite_score(
            stats,
            self.stage_config.competence,
            self.stage_config.composite_scoring,
        )

        return stats


    def check_promotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        thresholds = self.competence_thresholds
        base_thresholds = self.stage_config.competence
        stats = self.get_rolling_stats()

        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
            "thresholds_relaxed": thresholds != base_thresholds,
        }

        all_passed = True


        passed = self.stage_episodes >= thresholds.min_episodes
        results["checks"]["min_episodes"] = {
            "required": thresholds.min_episodes,
            "actual": self.stage_episodes,
            "passed": passed,
        }
        all_passed = all_passed and passed

        passed = self.stage_timesteps >= thresholds.min_timesteps
        results["checks"]["min_timesteps"] = {
            "required": thresholds.min_timesteps,
            "actual": self.stage_timesteps,
            "passed": passed,
        }
        all_passed = all_passed and passed


        min_required = max(MIN_EVALUATION_EPISODES, min(thresholds.evaluation_window, max(1, thresholds.min_episodes // 2)))
        if stats.window_size < min_required:
            results["checks"]["insufficient_data"] = {
                "required": min_required,
                "actual": stats.window_size,
                "passed": False,
            }
            results["promotion_ready"] = False
            results["stats"] = stats.to_dict()
            return False, results


        passed = stats.mean_profit_factor >= thresholds.min_profit_factor
        results["checks"]["profit_factor"] = {
            "required": thresholds.min_profit_factor,
            "actual": stats.mean_profit_factor,
            "passed": passed,
        }
        all_passed = all_passed and passed

        passed = stats.mean_drawdown <= thresholds.max_avg_drawdown
        results["checks"]["avg_drawdown"] = {
            "required": thresholds.max_avg_drawdown,
            "actual": stats.mean_drawdown,
            "passed": passed,
        }
        all_passed = all_passed and passed

        passed = stats.mean_pnl >= thresholds.min_avg_pnl
        results["checks"]["avg_pnl"] = {
            "required": thresholds.min_avg_pnl,
            "actual": stats.mean_pnl,
            "passed": passed,
        }
        all_passed = all_passed and passed


        if thresholds.min_avg_r_multiple > 0:
            passed = stats.mean_r_multiple >= thresholds.min_avg_r_multiple
            results["checks"]["r_multiple"] = {
                "required": thresholds.min_avg_r_multiple,
                "actual": stats.mean_r_multiple,
                "passed": passed,
            }
            all_passed = all_passed and passed


        passed_mean = stats.mean_win_rate >= thresholds.min_win_rate
        results["checks"]["win_rate_mean"] = {
            "required": thresholds.min_win_rate,
            "actual": stats.mean_win_rate,
            "passed": passed_mean,
        }


        wilson_required = getattr(thresholds, "min_win_rate_wilson_low", None)
        if wilson_required is None:
            wilson_required = max(0.0, float(thresholds.min_win_rate) - 0.05)


        min_trades_for_wilson = getattr(thresholds, "min_trades_for_wilson_gate", MIN_TRADES_FOR_WILSON_GATE_DEFAULT)
        pooled_trades_for_wilson = max(stats.total_wins + stats.total_losses, 0)
        if pooled_trades_for_wilson < int(min_trades_for_wilson):
            passed_wilson = True
            wilson_note = f"Skipped Wilson gate (pooled_trades={pooled_trades_for_wilson} < {int(min_trades_for_wilson)})"
        else:
            passed_wilson = stats.win_rate_wilson_low >= float(wilson_required)
            wilson_note = "Trade-level 95% Wilson lower bound"

        results["checks"]["win_rate_wilson_low"] = {
            "required": float(wilson_required),
            "actual": stats.win_rate_wilson_low,
            "passed": passed_wilson,
            "note": wilson_note,
            "pooled_trades": pooled_trades_for_wilson,
        }

        max_wilson_width = float(getattr(thresholds, "max_win_rate_wilson_width", 0.0) or 0.0)
        if max_wilson_width > 0.0:
            if pooled_trades_for_wilson < int(min_trades_for_wilson):
                passed_wilson_width = True
                wilson_width_note = (
                    f"Skipped Wilson width gate (pooled_trades={pooled_trades_for_wilson} < {int(min_trades_for_wilson)})"
                )
            else:
                passed_wilson_width = stats.win_rate_wilson_width <= max_wilson_width
                wilson_width_note = "Trade-level 95% Wilson interval width"

            results["checks"]["win_rate_wilson_width"] = {
                "required": float(max_wilson_width),
                "actual": stats.win_rate_wilson_width,
                "passed": passed_wilson_width,
                "note": wilson_width_note,
                "pooled_trades": pooled_trades_for_wilson,
            }
        else:
            passed_wilson_width = True

        all_passed = all_passed and passed_mean and passed_wilson and passed_wilson_width


        min_trades_ep = int(getattr(thresholds, "min_trades_per_episode_for_win_rate_stability", 0) or 0)
        eligible_eps = int(getattr(stats, "win_rate_stability_eligible_episodes", 0) or 0)
        eligible_trades = int(getattr(stats, "win_rate_stability_eligible_trades", 0) or 0)

        stability_std = (
            float(getattr(stats, "std_win_rate_trade_weighted", stats.std_win_rate))
            if eligible_eps >= 2
            else float(stats.std_win_rate)
        )
        passed = stability_std <= thresholds.max_win_rate_std
        results["checks"]["win_rate_stability"] = {
            "required": thresholds.max_win_rate_std,
            "actual": stability_std,
            "passed": passed,
            "std_unweighted": float(stats.std_win_rate),
            "std_trade_weighted": float(getattr(stats, "std_win_rate_trade_weighted", stats.std_win_rate)),
            "eligible_episodes": eligible_eps,
            "eligible_trades": eligible_trades,
            "min_trades_per_episode": min_trades_ep,
        }
        all_passed = all_passed and passed

        passed = stats.std_pnl <= thresholds.max_pnl_std
        results["checks"]["pnl_stability"] = {
            "required": thresholds.max_pnl_std,
            "actual": stats.std_pnl,
            "passed": passed,
        }
        all_passed = all_passed and passed

        # Activity is evidence, not a quota.  A selective policy may trade below
        # the per-episode target, but it only receives a waiver when edge is
        # distributed across enough eligible episodes/trades, the 95% episode
        # PnL lower bound is positive, and no drawdown breach occurred.  The old
        # `30 trades + mean_pnl > 0` rule was gameable by ten tiny winning
        # episodes among forty flat ones.
        enough_trades = stats.mean_trade_count >= thresholds.min_trade_count_avg
        selective_min_episodes = int(MIN_EVALUATION_EPISODES)
        selective_min_trades = max(
            int(MIN_TRADES_FOR_WILSON_GATE_DEFAULT),
            selective_min_episodes * max(1, min_trades_ep),
        )
        selective_edge_evidence = (
            eligible_eps >= selective_min_episodes
            and eligible_trades >= selective_min_trades
            and float(getattr(stats, "pnl_mean_ci_low", float("-inf"))) > 0.0
            and float(getattr(stats, "dd_breach_rate", 1.0)) == 0.0
        )

        passed = enough_trades or selective_edge_evidence
        results["checks"]["trade_activity"] = {
            "required_mean": thresholds.min_trade_count_avg,
            "actual_mean": stats.mean_trade_count,
            "passed": passed,
            "selective_evidence": bool(selective_edge_evidence),
            "eligible_episodes": eligible_eps,
            "required_eligible_episodes": selective_min_episodes,
            "eligible_trades": eligible_trades,
            "required_eligible_trades": selective_min_trades,
            "pnl_mean_ci_low": float(getattr(stats, "pnl_mean_ci_low", float("-inf"))),
            "dd_breach_rate": float(getattr(stats, "dd_breach_rate", 1.0)),
        }
        all_passed = all_passed and passed


        if getattr(thresholds, "min_avg_bars_between_trades", 0.0) > 0:
            passed = stats.mean_bars_between_trades >= thresholds.min_avg_bars_between_trades
            results["checks"]["avg_bars_between_trades"] = {
                "required": thresholds.min_avg_bars_between_trades,
                "actual": stats.mean_bars_between_trades,
                "passed": passed,
            }
            all_passed = all_passed and passed

        if getattr(thresholds, "min_setup_skipped_per_episode", 0.0) > 0:
            passed = stats.mean_setup_skipped_per_episode >= thresholds.min_setup_skipped_per_episode
            results["checks"]["setup_skipped_per_episode"] = {
                "required": thresholds.min_setup_skipped_per_episode,
                "actual": stats.mean_setup_skipped_per_episode,
                "passed": passed,
            }
            all_passed = all_passed and passed

        if getattr(thresholds, "min_entry_certainty_avg", 0.0) > 0:
            passed = stats.mean_entry_certainty >= thresholds.min_entry_certainty_avg
            results["checks"]["entry_certainty_avg"] = {
                "required": thresholds.min_entry_certainty_avg,
                "actual": stats.mean_entry_certainty,
                "passed": passed,
            }
            all_passed = all_passed and passed

        if getattr(thresholds, "min_avg_setup_quality", 0.0) > 0:
            passed = stats.mean_setup_quality >= thresholds.min_avg_setup_quality
            results["checks"]["avg_setup_quality"] = {
                "required": thresholds.min_avg_setup_quality,
                "actual": stats.mean_setup_quality,
                "passed": passed,
            }
            all_passed = all_passed and passed

        if getattr(thresholds, "max_fomo_trade_rate", 0.0) > 0:
            passed = stats.fomo_trade_rate <= thresholds.max_fomo_trade_rate
            results["checks"]["fomo_trade_rate"] = {
                "required": thresholds.max_fomo_trade_rate,
                "actual": stats.fomo_trade_rate,
                "passed": passed,
            }
            all_passed = all_passed and passed

        if getattr(thresholds, "max_revenge_trade_rate", 0.0) > 0:
            passed = stats.revenge_trade_rate <= thresholds.max_revenge_trade_rate
            results["checks"]["revenge_trade_rate"] = {
                "required": thresholds.max_revenge_trade_rate,
                "actual": stats.revenge_trade_rate,
                "passed": passed,
            }
            all_passed = all_passed and passed


        streak_required = int(getattr(thresholds, "consistency_streak_required", 0) or 0)
        streak_criteria = getattr(thresholds, "consistency_streak_criteria", {}) or {}
        if streak_required > 0 and streak_criteria:
            def _metric_from_ep(ep: EpisodeMetrics, key: str) -> float:
                if key == "win_rate":
                    return float(ep.win_rate)
                if key == "avg_bars_between_trades":
                    return float(ep.avg_bars_between_trades)
                if key == "entry_certainty_avg":
                    return float(ep.avg_entry_certainty)
                if key == "fomo_trade_rate":
                    return float(ep.fomo_trade_count) / max(int(ep.trade_count), 1)
                if key == "revenge_trade_rate":
                    return float(ep.revenge_trade_count) / max(int(ep.trade_count), 1)
                if key == "setup_skipped_per_episode":
                    return float(ep.setup_skipped_count)
                if key == "avg_setup_quality":
                    return float(ep.avg_setup_quality)
                return float(getattr(ep, key, 0.0) or 0.0)

            def _passes(ep: EpisodeMetrics, criteria: Dict[str, float]) -> bool:
                for k, v in criteria.items():
                    try:
                        actual = _metric_from_ep(ep, k)
                        threshold = float(v)
                    except Exception:
                        return False
                    if k in {"fomo_trade_rate", "revenge_trade_rate"}:
                        if actual > threshold:
                            return False
                    else:
                        if actual < threshold:
                            return False
                return True

            streak = 0
            for ep in reversed(self._history.get(self.current_stage, [])):
                if _passes(ep, streak_criteria):
                    streak += 1
                    if streak >= streak_required:
                        break
                else:
                    break

            passed = streak >= streak_required
            results["checks"]["consistency_streak"] = {
                "required": streak_required,
                "actual": streak,
                "passed": passed,
                "criteria": streak_criteria,
            }
            all_passed = all_passed and passed


        passed = stats.dd_breach_rate <= thresholds.max_dd_breach_rate
        results["checks"]["dd_breach_rate"] = {
            "required": thresholds.max_dd_breach_rate,
            "actual": stats.dd_breach_rate,
            "passed": passed,
        }
        all_passed = all_passed and passed

        passed = stats.consecutive_loss_breach_rate <= thresholds.max_consecutive_loss_rate
        results["checks"]["consecutive_loss_rate"] = {
            "required": thresholds.max_consecutive_loss_rate,
            "actual": stats.consecutive_loss_breach_rate,
            "passed": passed,
        }
        all_passed = all_passed and passed


        passed = stats.mask_collapse_rate <= thresholds.max_mask_collapse_rate
        results["checks"]["mask_collapse_rate"] = {
            "required": thresholds.max_mask_collapse_rate,
            "actual": stats.mask_collapse_rate,
            "passed": passed,
            "note": "Fraction of steps with HOLD-only action mask (flat/no pending entry)",
        }
        all_passed = all_passed and passed

        passed = stats.stop_mode_rate <= thresholds.max_stop_mode_rate
        results["checks"]["stop_mode_rate"] = {
            "required": thresholds.max_stop_mode_rate,
            "actual": stats.stop_mode_rate,
            "passed": passed,
            "note": "Fraction of steps in loss-layer stop-mode while flat",
        }
        all_passed = all_passed and passed

        entropy_targets = self.stage_config.entropy_targets
        min_entropy_samples = int(stats.window_size * 0.8)
        has_sufficient_entropy_samples = stats.entropy_samples >= min_entropy_samples

        if entropy_targets.use_in_promotion and entropy_targets.min_entropy > 0 and stats.mean_entropy >= 0:
            if has_sufficient_entropy_samples:
                passed = stats.mean_entropy >= entropy_targets.min_entropy
            else:

                passed = True
            results["checks"]["entropy"] = {
                "required": entropy_targets.min_entropy,
                "actual": stats.mean_entropy,
                "passed": passed,
                "entropy_samples": stats.entropy_samples,
                "min_samples_required": min_entropy_samples,
                "note": "Skipped due to insufficient samples" if not has_sufficient_entropy_samples else "Using EntropyTargets.min_entropy",
            }
            all_passed = all_passed and passed


        skill_reqs = self.stage_config.skill_requirements
        if skill_reqs.required_skills and self._skill_assessment:
            skill_passed, skill_results = self._skill_assessment.check_requirements(skill_reqs)
            results["checks"]["skills"] = skill_results
            all_passed = all_passed and skill_passed


        if self._composite_score and self.stage_config.composite_scoring.enabled:

            if thresholds != base_thresholds:
                effective_composite = compute_composite_score(
                    stats,
                    thresholds,
                    self.stage_config.composite_scoring,
                )
                results["composite_score"] = effective_composite.to_dict()
                results["composite_score_base"] = self._composite_score.to_dict()
                composite_passed = effective_composite.promotion_ready
                composite_actual = effective_composite.total_score
                composite_meets_floors = effective_composite.meets_hard_floors
            else:

                results["composite_score"] = self._composite_score.to_dict()
                composite_passed = self._composite_score.promotion_ready
                composite_actual = self._composite_score.total_score
                composite_meets_floors = self._composite_score.meets_hard_floors

            results["checks"]["composite_score"] = {
                "required": self.stage_config.composite_scoring.promotion_threshold,
                "actual": composite_actual,
                "passed": composite_passed,
                "meets_hard_floors": composite_meets_floors,
            }
            all_passed = all_passed and composite_passed


        gaming_results = self._anti_gaming_checker.run_all_checks(stats.to_dict())
        aggregate_gaming, gaming_concerns = self._anti_gaming_checker.get_aggregate_gaming_score(gaming_results)

        results["anti_gaming"] = {
            "aggregate_score": aggregate_gaming,
            "concerns": gaming_concerns,
            "checks": {name: r.gaming_score for name, r in gaming_results.items()},
        }


        if aggregate_gaming > 0.6:
            results["checks"]["anti_gaming"] = {
                "required": "< 0.6",
                "actual": aggregate_gaming,
                "passed": False,
                "concerns": gaming_concerns,
            }
            all_passed = False
        elif aggregate_gaming > 0.3:

            results["checks"]["anti_gaming"] = {
                "required": "< 0.6",
                "actual": aggregate_gaming,
                "passed": True,
                "warning": True,
                "concerns": gaming_concerns,
            }


        stage_idx = _stage_to_index(self.current_stage)
        if stage_idx >= 4:
            self._update_regime_assessment()
            if self._regime_assessment is not None:
                regime_score = self._regime_assessment.adaptation_score
                regime_coverage = self._regime_assessment.regime_coverage

                results["regime_assessment"] = {
                    "adaptation_score": regime_score,
                    "regime_coverage": regime_coverage,
                    "volatility_handling": self._regime_assessment.volatility_handling,
                    "trend_following": self._regime_assessment.trend_following,
                    "session_awareness": self._regime_assessment.session_awareness,
                    "cost_resilience": self._regime_assessment.cost_resilience,
                    "total_trades": self._regime_assessment.total_trades,
                    "confidence": self._regime_assessment.confidence,
                }


                min_adaptation = 0.35 if stage_idx >= 6 else 0.25
                if regime_score < min_adaptation:
                    results["checks"]["regime_adaptation"] = {
                        "required": min_adaptation,
                        "actual": regime_score,
                        "passed": False,
                        "reason": "Performance too inconsistent across market regimes",
                    }
                    all_passed = False
                else:
                    results["checks"]["regime_adaptation"] = {
                        "required": min_adaptation,
                        "actual": regime_score,
                        "passed": True,
                    }


                min_coverage = 0.5 if stage_idx >= 6 else 0.3
                if regime_coverage < min_coverage:
                    results["checks"]["regime_coverage"] = {
                        "required": min_coverage,
                        "actual": regime_coverage,
                        "passed": False,
                        "reason": "Not trading across enough market conditions",
                    }
                    all_passed = False
                else:
                    results["checks"]["regime_coverage"] = {
                        "required": min_coverage,
                        "actual": regime_coverage,
                        "passed": True,
                    }


                if stage_idx >= 6:
                    cost_resilience = self._regime_assessment.cost_resilience
                    vol_handling = self._regime_assessment.volatility_handling


                    min_stress_score = 0.35
                    stress_score = (cost_resilience + vol_handling) / 2

                    results["stress_resilience"] = {
                        "cost_resilience": cost_resilience,
                        "volatility_handling": vol_handling,
                        "combined_score": stress_score,
                        "min_required": min_stress_score,
                    }

                    if stress_score < min_stress_score:
                        results["checks"]["stress_resilience"] = {
                            "required": min_stress_score,
                            "actual": stress_score,
                            "passed": False,
                            "reason": "Poor performance under stress conditions (wide spreads, high volatility)",
                        }
                        all_passed = False
                    else:
                        results["checks"]["stress_resilience"] = {
                            "required": min_stress_score,
                            "actual": stress_score,
                            "passed": True,
                        }
            else:
                results["regime_assessment"] = {"status": "insufficient_data"}


        streak_required = int(getattr(thresholds, "consistency_streak_required", 0) or 0)
        streak_criteria = getattr(thresholds, "consistency_streak_criteria", {}) or {}
        if streak_required > 0 and streak_criteria:
            recent_eps = self._current_epoch_window(max(streak_required, 1))
            streak = 0
            for ep in reversed(recent_eps):
                meets = True
                for k, v in streak_criteria.items():
                    try:
                        key = str(k).lower()
                        req = float(v)
                        if key in ("win_rate",):
                            meets = meets and (float(ep.win_rate) >= req)
                        elif key in ("avg_bars_between_trades",):
                            meets = meets and (float(getattr(ep, "avg_bars_between_trades", 0.0)) >= req)
                        elif key in ("entry_certainty_avg",):
                            meets = meets and (float(getattr(ep, "avg_entry_certainty", 0.0)) >= req)
                        elif key in ("avg_setup_quality",):
                            meets = meets and (float(getattr(ep, "avg_setup_quality", 0.0)) >= req)
                        elif key in ("setup_skipped_per_episode",):
                            meets = meets and (float(getattr(ep, "setup_skipped_count", 0.0)) >= req)
                        elif key in ("fomo_trades", "fomo_trade_count"):
                            meets = meets and (float(getattr(ep, "fomo_trade_count", 0.0)) <= req)
                        elif key in ("revenge_trades", "revenge_trade_count"):
                            meets = meets and (float(getattr(ep, "revenge_trade_count", 0.0)) <= req)
                    except Exception:
                        continue
                if meets:
                    streak += 1
                else:
                    break

            passed = streak >= streak_required
            results["checks"]["consistency_streak"] = {
                "required": streak_required,
                "actual": streak,
                "passed": passed,
                "criteria": streak_criteria,
            }
            all_passed = all_passed and passed

        results["promotion_ready"] = all_passed
        results["stats"] = stats.to_dict()

        return all_passed, results

    def check_demotion_criteria(self) -> Tuple[bool, Dict[str, Any]]:
        config = self.stage_config
        if not config.allow_demotion:
            return False, {"reason": "demotion_disabled"}


        if _stage_to_index(self.current_stage) == 0:
            return False, {"reason": "at_foundation"}

        thresholds = self.stage_config.competence
        stats = self.get_rolling_stats()

        results: Dict[str, Any] = {
            "stage": self.current_stage.name,
            "stage_epoch": self._current_stage_epoch,
            "evaluation_window": stats.window_size,
            "checks": {},
        }


        min_episodes_for_demotion = max(50, thresholds.evaluation_window // 2)
        if stats.window_size < min_episodes_for_demotion:
            results["should_demote"] = False
            results["reason"] = "insufficient_data"
            results["stats"] = stats.to_dict()
            return False, results


        if self._composite_score and config.composite_scoring.enabled:
            if self._composite_score.demotion_risk:
                results["should_demote"] = True
                results["reason"] = "composite_score_below_threshold"
                results["composite_score"] = self._composite_score.to_dict()
                results["stats"] = stats.to_dict()
                return True, results


        demotion_factor = 0.70
        critical_failures = 0
        failure_reasons: List[str] = []

        if stats.mean_win_rate < thresholds.min_win_rate * demotion_factor:
            critical_failures += 1
            failure_reasons.append("win_rate_critical")
            results["checks"]["win_rate_critical"] = True

        if stats.mean_profit_factor < thresholds.min_profit_factor * demotion_factor:
            critical_failures += 1
            failure_reasons.append("profit_factor_critical")
            results["checks"]["profit_factor_critical"] = True

        if stats.mean_drawdown > thresholds.max_avg_drawdown * (1.0 / demotion_factor):
            critical_failures += 1
            failure_reasons.append("drawdown_critical")
            results["checks"]["drawdown_critical"] = True

        if stats.dd_breach_rate > thresholds.max_dd_breach_rate * (1.0 / demotion_factor):
            critical_failures += 1
            failure_reasons.append("dd_breach_critical")
            results["checks"]["dd_breach_critical"] = True


        stage_idx = _stage_to_index(self.current_stage)
        if stage_idx >= 4:
            if stats.fomo_trade_rate > 0.25:
                critical_failures += 1
                failure_reasons.append("excessive_fomo_trading")
                results["checks"]["excessive_fomo_trading"] = True
            if stats.revenge_trade_rate > 0.20:
                critical_failures += 1
                failure_reasons.append("revenge_trading")
                results["checks"]["revenge_trading"] = True
            if stats.mean_bars_between_trades < 2.0:
                critical_failures += 1
                failure_reasons.append("overtrading")
                results["checks"]["overtrading"] = True

        should_demote = critical_failures >= 2

        results["critical_failures"] = critical_failures
        results["failure_reasons"] = failure_reasons
        results["should_demote"] = should_demote
        results["stats"] = stats.to_dict()

        return should_demote, results

    def _internal_validation_check(self, promotion_results: Dict[str, Any]) -> Dict[str, Any]:
        stats = self.get_rolling_stats()
        val_cfg = self.stage_config.validation
        stage_idx = _stage_to_index(self.current_stage)

        result: Dict[str, Any] = {
            "type": "internal_stability_check",
            "passed": True,
            "checks": {},
        }


        recent_window = min(50, stats.window_size)
        recent_episodes = self._current_epoch_window(recent_window)

        if len(recent_episodes) >= 20:
            recent_wr = np.array([_clamp(_safe_float(ep.win_rate, 0.0), 0.0, 1.0) for ep in recent_episodes], dtype=np.float64)
            recent_trades = np.array([max(0.0, float(getattr(ep, "trade_count", 0))) for ep in recent_episodes], dtype=np.float64)

            min_trades_ep = int(
                getattr(self.stage_config.competence, "min_trades_per_episode_for_win_rate_stability", 0) or 0
            )
            if min_trades_ep < 0:
                min_trades_ep = 0
            eligible = (recent_trades >= float(min_trades_ep)) if min_trades_ep > 0 else (recent_trades > 0)
            eligible_n = int(np.sum(eligible))

            if eligible_n >= 2 and float(np.sum(recent_trades[eligible])) > 0.0:
                w = recent_trades[eligible]
                x = recent_wr[eligible]
                wsum = float(np.sum(w))
                mean = float(np.sum(w * x) / max(wsum, 1e-12))
                var = float(np.sum(w * (x - mean) ** 2) / max(wsum, 1e-12))
                wr_std = float(np.sqrt(max(var, 0.0)))
                wr_std_unweighted = float(np.std(x, ddof=1)) if eligible_n > 1 else 0.0
            else:
                wr_std = float(np.std(recent_wr))
                wr_std_unweighted = wr_std

            max_wr_std = 0.15 if stage_idx >= 6 else 0.20

            wr_stable = wr_std <= max_wr_std
            result["checks"]["win_rate_stability"] = {
                "std": wr_std,
                "max_allowed": max_wr_std,
                "passed": wr_stable,
                "std_unweighted": float(wr_std_unweighted),
                "eligible_episodes": int(eligible_n),
                "min_trades_per_episode": int(min_trades_ep),
            }
            if not wr_stable:
                result["passed"] = False


        if len(recent_episodes) >= 20 and stats.window_size >= 50:
            recent_mean_wr = float(np.mean([ep.win_rate for ep in recent_episodes]))
            overall_mean_wr = stats.mean_win_rate


            min_ratio = val_cfg.min_performance_ratio if hasattr(val_cfg, 'min_performance_ratio') else 0.85
            if overall_mean_wr > 0.01:
                perf_ratio = recent_mean_wr / overall_mean_wr
                ratio_ok = perf_ratio >= min_ratio
                result["checks"]["performance_ratio"] = {
                    "recent_wr": recent_mean_wr,
                    "overall_wr": overall_mean_wr,
                    "ratio": perf_ratio,
                    "min_required": min_ratio,
                    "passed": ratio_ok,
                }
                if not ratio_ok:
                    result["passed"] = False


        max_dd_breach_rate = 0.05 if stage_idx >= 6 else 0.10
        dd_breach_ok = stats.dd_breach_rate <= max_dd_breach_rate
        result["checks"]["safety"] = {
            "dd_breach_rate": stats.dd_breach_rate,
            "max_allowed": max_dd_breach_rate,
            "passed": dd_breach_ok,
        }
        if not dd_breach_ok:
            result["passed"] = False


        if stage_idx >= 6 and self._regime_assessment is not None:
            adaptation = self._regime_assessment.adaptation_score
            min_adaptation = 0.4
            adapt_ok = adaptation >= min_adaptation
            result["checks"]["regime_stability"] = {
                "adaptation_score": adaptation,
                "min_required": min_adaptation,
                "passed": adapt_ok,
            }
            if not adapt_ok:
                result["passed"] = False

        return result


    def _validation_retry_interval(self, config: CurriculumStageConfig) -> int:
        """Return the minimum policy-development distance between gate attempts."""
        evaluation_window = max(
            0,
            int(getattr(getattr(config, "competence", None), "evaluation_window", 0) or 0),
        )
        return max(25, evaluation_window // 2)

    def _validation_attempt_is_due(self, config: CurriculumStageConfig) -> bool:
        if self._last_validation_attempt_stage != self.current_stage.name:
            return True
        if self._last_validation_attempt_epoch != self._current_stage_epoch:
            return True
        episodes_since_attempt = (
            int(self.stage_episodes) - int(self._last_validation_attempt_stage_episode)
        )
        return episodes_since_attempt >= self._validation_retry_interval(config)

    def _record_validation_attempt(self) -> None:
        self._last_validation_attempt_stage = self.current_stage.name
        self._last_validation_attempt_epoch = self._current_stage_epoch
        self._last_validation_attempt_stage_episode = self.stage_episodes

    def try_promote(
        self,
        *,
        readiness_only: bool = False,
    ) -> Tuple[bool, Optional[CurriculumStage]]:
        """Evaluate stage readiness and optionally perform the transition.

        ``readiness_only`` reuses the exact promotion, invariant, validation,
        and stress gates for terminal-stage mastery without attempting a stage
        transition.  It deliberately does not cache a successful result: a
        later call evaluates the then-current policy again.
        """
        if not self.auto_promote and not readiness_only:
            return False, None

        config = self.stage_config
        if config.is_terminal and not readiness_only:
            return False, None

        next_stage = get_next_stage(self.current_stage)
        if next_stage is None and not readiness_only:
            return False, None


        if self._transition_cooldown_remaining > 0:
            return False, None

        meets_criteria, results = self.check_promotion_criteria()
        if not meets_criteria:
            return False, None


        invariant_summary = self._invariant_checker.get_summary()
        critical_violations = invariant_summary.get("critical_count", 0)
        if critical_violations > 0:
            if self.verbose:
                logger.warning(
                    f"Blocking promotion due to {critical_violations} critical invariant violations. "
                    f"Details: {invariant_summary}"
                )
            results["invariant_block"] = {
                "blocked": True,
                "critical_count": critical_violations,
                "summary": invariant_summary,
            }
            return False, None


        gate_config = getattr(config, "validation", None)
        if gate_config is not None and getattr(gate_config, "enabled", False):
            if not self._validation_attempt_is_due(config):
                retry_interval = self._validation_retry_interval(config)
                episodes_since_attempt = (
                    self.stage_episodes - self._last_validation_attempt_stage_episode
                )
                results["validation_retry"] = {
                    "due": False,
                    "retry_interval_episodes": retry_interval,
                    "episodes_since_attempt": episodes_since_attempt,
                    "episodes_until_retry": max(0, retry_interval - episodes_since_attempt),
                }
                return False, None

            # Record before invoking user-provided evaluators so exceptions are
            # also bounded rather than retried on every subsequent episode.
            self._record_validation_attempt()

            if self.validation_gate_evaluator is not None:
                try:
                    stage_idx = _stage_to_index(self.current_stage)

                    scenarios = self._validation_gate.get_scenarios_for_stage(stage_idx)
                    scenario_payload = [
                        s.to_dict() if hasattr(s, "to_dict") else {"name": s.name} for s in scenarios
                    ]


                    stats = self.get_rolling_stats()
                    training_stats = {
                        "mean_win_rate": stats.mean_win_rate,
                        "mean_profit_factor": stats.mean_profit_factor,
                        "mean_r_multiple": stats.mean_r_multiple,
                        "mean_pnl": stats.mean_pnl,
                        "mean_trade_count": stats.mean_trade_count,
                        "total_trades": stats.total_trades,
                    }


                    validation_results = self.validation_gate_evaluator(scenario_payload, training_stats)


                    gate_result = self._validation_gate.evaluate_all(
                        validation_results=validation_results,
                        training_stats=training_stats,
                        stage_name=self.current_stage.name,
                        stage_epoch=self._current_stage_epoch,
                        stage_index=stage_idx,
                        scenarios=scenarios,
                    )

                    results["validation_gate"] = {
                        "passed": gate_result.gate_passed,
                        "pass_rate": gate_result.pass_rate,
                        "scenarios_passed": gate_result.scenarios_passed,
                        "scenarios_total": gate_result.scenarios_total,
                        "performance_ratio": gate_result.performance_ratio,
                        "blocking_reasons": gate_result.blocking_reasons,
                        "recommendations": self._validation_gate.get_recommendations(),
                    }


                    self._validation_gate_history.append({
                        "stage": self.current_stage.name,
                        "stage_epoch": self._current_stage_epoch,
                        "stage_episodes": self.stage_episodes,
                        "timestamp": _now_iso(self.tz),
                        "passed": gate_result.gate_passed,
                        "pass_rate": gate_result.pass_rate,
                        "scenarios_passed": gate_result.scenarios_passed,
                        "scenarios_total": gate_result.scenarios_total,
                        "performance_ratio": gate_result.performance_ratio,
                        "blocking_reasons": gate_result.blocking_reasons,
                    })

                    if len(self._validation_gate_history) > 100:
                        self._validation_gate_history = self._validation_gate_history[-100:]

                    if not gate_result.gate_passed:
                        if self.verbose:
                            logger.info(
                                f"Validation gate failed: {gate_result.scenarios_passed}/{gate_result.scenarios_total} "
                                f"scenarios passed. Blocking promotion. Reasons: {gate_result.blocking_reasons}"
                            )
                        return False, None

                except Exception as e:
                    logger.warning(f"Validation gate evaluator error: {e}; blocking promotion.")
                    results["validation_gate"] = {"error": str(e)}
                    return False, None
            else:

                results["validation_gate_pending"] = True
                if self.verbose:
                    logger.debug("Validation gate enabled but no evaluator provided - skipping")
                if readiness_only and self.validation_evaluator is None:
                    # Terminal mastery is a development-holdout claim.  It may
                    # not fall back to training-window statistics when no
                    # external evaluator is wired.
                    return False, None


        stress_config = getattr(config, "stress_test", None)
        if stress_config is not None and getattr(stress_config, "enabled", False):
            if self.stress_test_evaluator is not None:
                try:

                    scenarios = self._stress_tester.get_stress_scenarios()


                    stats = self.get_rolling_stats()
                    baseline_stats = {
                        "mean_win_rate": stats.mean_win_rate,
                        "mean_profit_factor": stats.mean_profit_factor,
                        "mean_r_multiple": stats.mean_r_multiple,
                        "mean_pnl": stats.mean_pnl,
                    }


                    stress_episode_results = self.stress_test_evaluator(scenarios, baseline_stats)


                    stress_results = []
                    for i, scenario in enumerate(scenarios):
                        if i < len(stress_episode_results):
                            eps = stress_episode_results[i]
                            result = self._stress_tester.evaluate_stress_result(
                                scenario, eps if isinstance(eps, list) else [eps], baseline_stats
                            )
                            stress_results.append(result)


                    robustness_score = self._stress_tester.get_robustness_score(stress_results)
                    min_robustness = float(getattr(stress_config, "min_robustness_score", 0.6))

                    results["stress_test"] = {
                        "passed": robustness_score >= min_robustness,
                        "robustness_score": robustness_score,
                        "min_required": min_robustness,
                        "summary": self._stress_tester.get_summary(stress_results),
                    }


                    self._stress_test_history.append({
                        "stage": self.current_stage.name,
                        "stage_epoch": self._current_stage_epoch,
                        "stage_episodes": self.stage_episodes,
                        "timestamp": _now_iso(self.tz),
                        "passed": robustness_score >= min_robustness,
                        "robustness_score": robustness_score,
                        "min_required": min_robustness,
                        "scenarios_count": len(stress_results),
                        "summary": self._stress_tester.get_summary(stress_results),
                    })

                    if len(self._stress_test_history) > 100:
                        self._stress_test_history = self._stress_test_history[-100:]

                    if not results["stress_test"]["passed"]:
                        if self.verbose:
                            logger.info(
                                f"Stress test failed: robustness {robustness_score:.2%} < {min_robustness:.2%}. "
                                f"Blocking promotion."
                            )
                        return False, None

                except Exception as e:
                    logger.warning(f"Stress test evaluator error: {e}; blocking readiness.")
                    results["stress_test"] = {"error": str(e)}
                    return False, None
            else:

                results["stress_test_pending"] = True
                if self.verbose:
                    logger.debug("Stress test enabled but no evaluator provided - skipping")
                if readiness_only:
                    return False, None


        val_cfg = self.stage_config.validation
        if val_cfg.enabled and self.validation_gate_evaluator is None:
            if self.validation_evaluator is None:

                val_result = self._internal_validation_check(results)
                results["validation"] = val_result
                if not val_result.get("passed", False):
                    if self.verbose:
                        logger.info(f"Internal validation failed; blocking promotion. Details: {val_result}")
                    return False, None
            else:
                try:
                    val_result = self.validation_evaluator(self.stage_config)

                    results["validation"] = val_result
                    if not val_result.get("passed", False):
                        if self.verbose:
                            logger.info(f"Validation failed; blocking promotion. Details: {val_result}")
                        return False, None
                except Exception as e:
                    logger.warning(f"Validation evaluator error: {e}; blocking readiness.")
                    results["validation"] = {"error": str(e), "passed": False}
                    return False, None


        if self.current_stage == CurriculumStage.STRATEGIST and next_stage == CurriculumStage.PROFESSIONAL:

            if self._selectivity_phase_active:
                return False, None


            if self._selectivity_phase_completed_epoch != self._current_stage_epoch:
                self.start_selectivity_phase(episodes=20)
                results["selectivity_phase_started"] = True
                return False, None

        if readiness_only:
            return True, self.current_stage

        # The non-transitioning readiness path above is the only valid route
        # with no successor (the terminal stage).  Keep the mutating path
        # explicitly narrowed for both runtime safety and static analysis.
        if next_stage is None:
            return False, None

        old_stage = self.current_stage
        self.current_stage = next_stage
        self._enter_stage(next_stage, reason="promotion")

        transition_info = {
            "type": "promotion",
            "from_stage": old_stage.name,
            "to_stage": next_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
        }
        self._transitions.append(transition_info)

        if self.verbose:
            logger.info(f"🎓 PROMOTION: {old_stage.name} → {next_stage.name} (epoch={self._current_stage_epoch})")


        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("promotion", old_stage, next_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")

        return True, next_stage

    def try_demote(self) -> Tuple[bool, Optional[CurriculumStage]]:
        if not self.auto_demote:
            return False, None

        previous_stage = get_previous_stage(self.current_stage)
        if previous_stage is None:
            return False, None


        if self._transition_cooldown_remaining > 0:
            return False, None

        should_demote, results = self.check_demotion_criteria()
        if not should_demote:
            return False, None

        old_stage = self.current_stage


        stats = self.get_rolling_stats()
        failure_reasons = results.get("failure_reasons", [])
        self._demotion_analyzer.record_demotion(
            from_stage=old_stage,
            to_stage=previous_stage,
            failure_reasons=failure_reasons,
            skill_assessment=self._skill_assessment,
            stats=stats,
            global_episode=self.total_episodes,
        )

        self.current_stage = previous_stage

        self._enter_stage(previous_stage, reason="demotion", demoted_from_stage=old_stage)

        transition_info = {
            "type": "demotion",
            "from_stage": old_stage.name,
            "to_stage": previous_stage.name,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "criteria_results": results,
            "failure_reasons": failure_reasons,
        }
        self._transitions.append(transition_info)

        if self.verbose:
            logger.warning(f"📉 DEMOTION: {old_stage.name} → {previous_stage.name} (epoch={self._current_stage_epoch})")


        if self.on_transition_callback is not None:
            try:
                self.on_transition_callback("demotion", old_stage, previous_stage, transition_info)
            except Exception as e:
                logger.warning(f"Transition callback error: {e}")

        return True, previous_stage

    def update(self) -> Tuple[bool, Optional[CurriculumStage]]:
        promoted, new_stage = self.try_promote()
        if promoted:
            return True, new_stage

        demoted, new_stage = self.try_demote()
        if demoted:
            return True, new_stage

        return False, None


    def check_and_maybe_promote(self) -> Tuple[bool, Dict[str, Any]]:
        promotion_ready, results = self.check_promotion_criteria()
        if not isinstance(results, dict):
            results = {}
        results.setdefault("promotion_ready", bool(promotion_ready))

        if not promotion_ready:
            return False, results

        promoted, _ = self.try_promote()
        if promoted and self._transitions:
            last = self._transitions[-1]
            if isinstance(last, dict) and last.get("type") == "promotion":
                crit = last.get("criteria_results")
                if isinstance(crit, dict):
                    return True, crit

        return bool(promoted), results

    def check_and_maybe_demote(self) -> Tuple[bool, Dict[str, Any]]:
        should_demote, results = self.check_demotion_criteria()
        if not isinstance(results, dict):
            results = {}
        results.setdefault("should_demote", bool(should_demote))

        if not should_demote:
            return False, results

        demoted, _ = self.try_demote()
        if demoted and self._transitions:
            last = self._transitions[-1]
            if isinstance(last, dict) and last.get("type") == "demotion":
                crit = last.get("criteria_results")
                if isinstance(crit, dict):
                    return True, crit

        return bool(demoted), results

    def force_stage(self, stage: CurriculumStage, reason: str = "manual") -> None:
        old_stage = self.current_stage
        self.current_stage = stage
        self._enter_stage(stage, reason=f"force:{reason}")

        self._transitions.append({
            "type": "force",
            "from_stage": old_stage.name,
            "to_stage": stage.name,
            "reason": reason,
            "timestamp": _now_iso(self.tz),
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
        })

        if self.verbose:
            logger.info(f"⚡ FORCE STAGE: {old_stage.name} → {stage.name} (reason={reason})")


    def _get_regime_assessment_report(self) -> Dict[str, Any]:
        self._update_regime_assessment()

        if self._regime_assessment is None:
            return {
                "status": "insufficient_data",
                "total_trades": len(self._regime_trades),
                "min_required": 50,
            }

        ra = self._regime_assessment

        return {
            "status": "active",
            "total_trades": ra.total_trades,
            "confidence": ra.confidence,


            "scores": {
                "adaptation": ra.adaptation_score,
                "volatility_handling": ra.volatility_handling,
                "trend_following": ra.trend_following,
                "session_awareness": ra.session_awareness,
                "cost_resilience": ra.cost_resilience,
            },


            "regime_coverage": ra.regime_coverage,


            "weaknesses": ra.get_weaknesses(threshold=0.4),


            "volatility_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.volatility_performance.items()
            },
            "trend_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.trend_performance.items()
            },
            "session_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.session_performance.items()
            },
            "spread_breakdown": {
                k.value: {
                    "trades": v.trade_count,
                    "win_rate": v.win_rate,
                    "avg_r": v.avg_r,
                }
                for k, v in ra.spread_performance.items()
            },
        }

    def get_progress_report(self) -> Dict[str, Any]:
        stats = self.get_rolling_stats()
        meets_promotion, promotion_results = self.check_promotion_criteria()
        meets_demotion, demotion_results = self.check_demotion_criteria()

        progression = get_stage_progression()
        stage_idx = progression.index(self.current_stage)


        reward_cfg = self.stage_config.rewards if self.stage_config else None
        bonuses_enabled = False
        if reward_cfg:

            bonuses_enabled = (
                getattr(reward_cfg, 'trailing_stop_bonus', 0.0) > 0.01 or
                getattr(reward_cfg, 'r_multiple_bonus_scale', 0.0) > 0.01 or
                getattr(reward_cfg, 'agent_close_bonus', 0.0) > 0.01
            )

        phase_info = {
            "phase_num": stage_idx,
            "phase_name": self.current_stage.name,
            "description": getattr(self.stage_config, 'description', '') if self.stage_config else '',
            "bonuses_enabled": bonuses_enabled,
            "loss_multiplier_range": f"{getattr(reward_cfg, 'loss_penalty_mult', 1.0):.1f}x" if reward_cfg else "1.0x",
            "market_difficulty": "Progressive" if stage_idx < 5 else "Full",
        }


        blockers = []
        for check_name, check_data in promotion_results.get("checks", {}).items():
            if isinstance(check_data, dict) and not check_data.get("passed", True):
                required = check_data.get("required", 0)
                actual = check_data.get("actual", 0)


                is_max_metric = any(kw in check_name.lower() for kw in [
                    "max_", "std", "breach", "drawdown", "loss", "collapse", "stop_mode"
                ])

                if is_max_metric:

                    gap = actual - required
                else:

                    gap = required - actual

                blockers.append({
                    "metric": check_name,
                    "gap": gap,
                    "required": required,
                    "actual": actual,
                    "is_max_metric": is_max_metric,
                })
        blockers.sort(key=lambda x: abs(x.get("gap", 0)), reverse=True)


        recommendations = self._generate_recommendations(blockers)


        patience_recs: List[str] = []
        if stats.mean_bars_between_trades < 3.0:
            patience_recs.append("Wait longer between trades - aim for at least 3 bars")
        if stats.fomo_trade_rate > 0.2:
            patience_recs.append("Reduce FOMO trading - wait for clear setups")
        if stats.revenge_trade_rate > 0.15:
            patience_recs.append("Avoid revenge trading - enforce cooldown after losses")
        if stats.mean_setup_skipped_per_episode < 2.0:
            patience_recs.append("Skip more marginal setups - increase selectivity")
        if stats.mean_entry_certainty < 0.6:
            patience_recs.append("Increase entry certainty - trade only high-confidence setups")

        if patience_recs:
            recommendations = list(dict.fromkeys(recommendations + patience_recs))


        estimated_episodes = 0
        if blockers and not meets_promotion:
            velocity = self._learning_velocity
            avg_improvement = velocity.get_average_improvement()
            if avg_improvement > 0:
                avg_gap = np.mean([abs(b["gap"]) for b in blockers if b["gap"] is not None])
                estimated_episodes = int(avg_gap / avg_improvement * 10)
            else:
                estimated_episodes = -1

        return {
            "version": STATE_VERSION,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "stage_index": stage_idx,
            "total_stages": len(progression),
            "progress_pct": (stage_idx / (len(progression) - 1)) * 100 if len(progression) > 1 else 100,


            "phase_info": phase_info,


            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps": self.stage_timesteps,
            "stage_episodes": self.stage_episodes,


            "rolling_stats": stats.to_dict(),
            "patience_metrics": {
                "mean_bars_between_trades": stats.mean_bars_between_trades,
                "std_bars_between_trades": stats.std_bars_between_trades,
                "mean_setup_skipped_per_episode": stats.mean_setup_skipped_per_episode,
                "mean_entry_certainty": stats.mean_entry_certainty,
                "fomo_trade_rate": stats.fomo_trade_rate,
                "revenge_trade_rate": stats.revenge_trade_rate,
                "patience_consistency": stats.patience_consistency,
            },


            "promotion_ready": meets_promotion,
            "promotion_checks": promotion_results.get("checks", {}),
            "promotion_blockers": blockers[:5],
            "demotion_risk": meets_demotion,
            "demotion_checks": demotion_results.get("checks", {}),


            "anti_gaming": promotion_results.get("anti_gaming", {}),


            "estimated_episodes_to_promotion": estimated_episodes,
            "recommendations": recommendations,


            "learning_velocity": self._learning_velocity.to_dict(),
            "is_plateaued": self._learning_velocity.is_plateaued(),
            "skill_assessment": self._skill_assessment.to_dict() if self._skill_assessment else None,
            "composite_score": self._composite_score.to_dict() if self._composite_score else None,
            "recovery_protocol": self._recovery_state.to_dict(),
            "review_session": self._review_state.to_dict(),
            "demotion_analysis": self._demotion_analyzer.to_dict(),
            "selectivity_phase": {
                "active": self._selectivity_phase_active,
                "episodes_remaining": (
                    self._selectivity_phase_episodes_remaining if self._selectivity_phase_active else 0
                ),
                "requirements": self._selectivity_phase_requirements if self._selectivity_phase_active else {},
                "completed_epoch": self._selectivity_phase_completed_epoch,
                "failures": self._selectivity_phase_failures,
            },


            "invariant_summary": self._invariant_checker.get_summary(),


            "regime_assessment": self._get_regime_assessment_report(),


            "is_in_transition": self.is_in_transition,
            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,
            "reward_blend_factor": self.reward_blend_factor,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_multiplier": self.get_lr_multiplier(),


            "transitions_count": len(self._transitions),
            "recent_transitions": self._transitions[-5:],


            "validation_gate_history": self._validation_gate_history[-10:],
            "last_validation_gate": self._validation_gate_history[-1] if self._validation_gate_history else None,


            "stress_test_history": self._stress_test_history[-10:],
            "last_stress_test": self._stress_test_history[-1] if self._stress_test_history else None,
        }

    def _generate_recommendations(self, blockers: List[Dict]) -> List[str]:
        recs = []

        for blocker in blockers[:3]:
            metric = blocker.get("metric", "")

            if "win_rate" in metric:
                recs.append("Focus on entry quality - wait for higher-confidence setups")
            elif "drawdown" in metric:
                recs.append("Reduce position sizes or tighten stops to control drawdown")
            elif "profit_factor" in metric:
                recs.append("Improve risk-reward: let winners run longer, cut losers faster")
            elif "stability" in metric or "std" in metric:
                recs.append("Maintain consistent approach across different market conditions")
            elif "trade" in metric and "activity" in metric:
                recs.append("Increase trading activity with quality entries")
            elif "dd_breach" in metric:
                recs.append("Critical: Avoid drawdown breaches - reduce risk per trade")
            elif "consecutive_loss" in metric:
                recs.append("Take breaks after consecutive losses to reset")
            elif "skill" in metric:
                if self._skill_assessment and self._skill_assessment.weakest_skills:
                    weak = self._skill_assessment.weakest_skills[0].value
                    recs.append(f"Focus on improving {weak.replace('_', ' ')}")


        if self.is_learning_plateaued():
            recs.append("Learning has plateaued - consider adjusting strategy or hyperparameters")


        return list(dict.fromkeys(recs))


    def should_stop_training(
        self,
        *,
        max_timesteps: Optional[int] = None,
        max_episodes: Optional[int] = None,
        max_hours: Optional[float] = None,
        start_time: Optional[float] = None,
        plateau_stop: bool = False,
        plateau_threshold_episodes: int = 500,
        max_demotions_from_same_stage: int = 5,
        mastery_confirmation_episodes: int = 100,
    ) -> Tuple[bool, str]:

        if self.stage_config.is_terminal:
            competence = self.stage_config.competence
            required_episodes = max(
                int(mastery_confirmation_episodes),
                int(getattr(competence, "min_episodes", 0) or 0),
            )
            required_timesteps = int(getattr(competence, "min_timesteps", 0) or 0)

            if (
                self.stage_episodes >= required_episodes
                and self.stage_timesteps >= required_timesteps
            ):
                # Reuse the promotion path in non-transitioning mode so LIVE_READY
                # must satisfy competence, invariants, and the configured
                # development validation/stress gates.  Reaching the enum value
                # alone is not evidence of mastery or real-capital readiness.
                mastery_ready, _ = self.try_promote(readiness_only=True)
                if mastery_ready:
                    meets_demotion, _ = self.check_demotion_criteria()
                    if not meets_demotion:
                        return (
                            True,
                            f"GOAL_ACHIEVED: Reached {self.current_stage.name}, "
                            f"met competence and validation gates, and maintained "
                            f"for {self.stage_episodes} episodes / "
                            f"{self.stage_timesteps:,} timesteps",
                        )


        if max_timesteps is not None and self.total_timesteps >= max_timesteps:
            return True, f"MAX_TIMESTEPS: Reached {self.total_timesteps:,} timesteps"

        if max_episodes is not None and self.total_episodes >= max_episodes:
            return True, f"MAX_EPISODES: Reached {self.total_episodes:,} episodes"

        if max_hours is not None and start_time is not None:
            import time
            elapsed_hours = (time.time() - start_time) / 3600.0
            if elapsed_hours >= max_hours:
                return True, f"MAX_HOURS: Training ran for {elapsed_hours:.1f} hours"


        if plateau_stop and self._learning_velocity.is_plateaued(plateau_threshold_episodes):
            return True, f"PLATEAU: No improvement for {self._learning_velocity.plateau_episodes} episodes"


        for stage, count in self._demotion_analyzer.stage_failure_counts.items():
            if count >= max_demotions_from_same_stage:
                return True, f"REPEATED_FAILURE: Demoted from {stage.name} {count} times"

        return False, "TRAINING"

    def get_training_status(self) -> Dict[str, Any]:
        progression = get_stage_progression()
        stage_idx = progression.index(self.current_stage)

        return {
            "current_stage": self.current_stage.name,
            "stage_index": stage_idx,
            "total_stages": len(progression),
            "progress_pct": (stage_idx / max(len(progression) - 1, 1)) * 100,
            "is_terminal": self.stage_config.is_terminal,
            "stage_episodes": self.stage_episodes,
            "total_episodes": self.total_episodes,
            "total_timesteps": self.total_timesteps,
            "plateau_episodes": self._learning_velocity.plateau_episodes,
            "is_plateaued": self._learning_velocity.is_plateaued(),
            "demotion_counts": {s.name: c for s, c in self._demotion_analyzer.stage_failure_counts.items()},
        }


    def to_dict(self) -> Dict[str, Any]:
        history_serialized: Dict[str, List[Dict[str, Any]]] = {}
        for stage, episodes in self._history.items():
            history_serialized[stage.name] = [asdict(ep) for ep in episodes]

        return {
            "version": STATE_VERSION,
            "tz": self.tz,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps_total": {s.name: t for s, t in self._stage_timesteps_total.items()},
            "stage_episodes_total": {s.name: e for s, e in self._stage_episodes_total.items()},
            "stage_epoch_counter": {s.name: e for s, e in self._stage_epoch_counter.items()},
            "stage_timesteps_current": self.stage_timesteps,
            "stage_episodes_current": self.stage_episodes,
            "history": history_serialized,
            "transitions": self._transitions,
            "demotion_analyzer": self._demotion_analyzer.to_dict(),
            "recovery_state": self._recovery_state.to_dict(),
            "review_state": self._review_state.to_dict(),
            "learning_velocity": {
                "plateau_episodes": self._learning_velocity.plateau_episodes,
                "improvement_rates": self._learning_velocity.improvement_rates,
                "metric_history": (
                    {k: list(v)[-100:] for k, v in self._learning_velocity.metric_history.items()}
                    if hasattr(self._learning_velocity, "metric_history")
                    else {}
                ),
            },

            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,

            "previous_stage": self._previous_stage_name,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_warmup_factor": self._lr_warmup_factor,

            "last_episode_end_idx": self._last_episode_end_idx,

            "processed_episode_ids": list(self._processed_episode_ids),

            "current_entropy": self._current_entropy,

            "review_tick_episode": self._review_tick_episode,

            "rng_state": self._rng.bit_generator.state,

            "validation_gate_history": self._validation_gate_history[-100:],

            "validation_retry": {
                "stage": self._last_validation_attempt_stage,
                "stage_epoch": self._last_validation_attempt_epoch,
                "stage_episode": self._last_validation_attempt_stage_episode,
            },

            "stress_test_history": self._stress_test_history[-100:],

            "selectivity_phase": {
                "active": self._selectivity_phase_active,
                "episodes_remaining": self._selectivity_phase_episodes_remaining,
                "requirements": self._selectivity_phase_requirements,
                "completed_epoch": self._selectivity_phase_completed_epoch,
                "target_episodes": self._selectivity_phase_target_episodes,
                "failures": self._selectivity_phase_failures,
            },
            "saved_at": _now_iso(self.tz),
        }

    @classmethod
    def from_dict(cls, state: Dict[str, Any], **kwargs: Any) -> "CurriculumManager":

        loaded_version = state.get("version", "1.0")
        if loaded_version != STATE_VERSION:
            logger.warning(
                f"Loading checkpoint from v{loaded_version} (current: v{STATE_VERSION}). "
                "New features (recovery_state, review_state, learning_velocity) may use defaults."
            )

        tz = state.get("tz", DEFAULT_TZ)
        current_stage = CurriculumStage[state["current_stage"]]

        manager = cls(initial_stage=current_stage, tz=tz, **kwargs)
        manager.total_timesteps = _safe_int(state.get("total_timesteps", 0), 0)
        manager.total_episodes = _safe_int(state.get("total_episodes", 0), 0)


        for stage_name, timesteps in (state.get("stage_timesteps_total", {}) or {}).items():
            try:
                manager._stage_timesteps_total[CurriculumStage[stage_name]] = _safe_int(timesteps, 0)
            except KeyError:
                logger.debug(f"Unknown stage in timesteps_total: {stage_name}")

        for stage_name, episodes in (state.get("stage_episodes_total", {}) or {}).items():
            try:
                manager._stage_episodes_total[CurriculumStage[stage_name]] = _safe_int(episodes, 0)
            except KeyError:
                logger.debug(f"Unknown stage in episodes_total: {stage_name}")


        for stage_name, epoch in (state.get("stage_epoch_counter", {}) or {}).items():
            try:
                manager._stage_epoch_counter[CurriculumStage[stage_name]] = _safe_int(epoch, 0)
            except KeyError:
                logger.debug(f"Unknown stage in epoch_counter: {stage_name}")

        manager._current_stage_epoch = _safe_int(
            state.get("current_stage_epoch", manager._current_stage_epoch),
            manager._current_stage_epoch,
        )
        manager.stage_timesteps = _safe_int(state.get("stage_timesteps_current", 0), 0)
        manager.stage_episodes = _safe_int(state.get("stage_episodes_current", 0), 0)


        progression = get_stage_progression()
        manager._history = {stage: deque(maxlen=manager.max_history_size) for stage in progression}
        for stage_name, episodes_data in (state.get("history", {}) or {}).items():
            try:
                stage = CurriculumStage[stage_name]
            except KeyError:
                logger.debug(f"Unknown stage in history: {stage_name}")
                continue
            for ep_data in (episodes_data or []):
                try:
                    manager._history[stage].append(EpisodeMetrics.from_dict(ep_data))
                except Exception as e:
                    logger.debug(f"Skipping malformed episode in history: {e}")
                    continue

        manager._transitions = state.get("transitions", []) or []


        demotion_data = state.get("demotion_analyzer", {})
        if demotion_data:
            manager._demotion_analyzer.load_from_dict(demotion_data)


        recovery_data = state.get("recovery_state", {})
        if recovery_data:
            manager._recovery_state = RecoveryProtocolState.from_dict(recovery_data)


        review_data = state.get("review_state", {})
        if review_data:
            manager._review_state = ReviewSessionState.from_dict(review_data)


        velocity_data = state.get("learning_velocity", {})
        if velocity_data:
            manager._learning_velocity.plateau_episodes = velocity_data.get("plateau_episodes", 0)
            manager._learning_velocity.improvement_rates = velocity_data.get("improvement_rates", {})
            metric_history = velocity_data.get("metric_history", {})
            if metric_history and hasattr(manager._learning_velocity, "metric_history"):
                for k, v in metric_history.items():
                    manager._learning_velocity.metric_history[k] = deque(v, maxlen=100)


        manager._transition_cooldown_remaining = _safe_int(state.get("transition_cooldown_remaining", 0), 0)
        manager._reward_blend_remaining = _safe_int(state.get("reward_blend_remaining", 0), 0)
        manager._lr_warmup_active = bool(state.get("lr_warmup_active", False))
        manager._lr_warmup_steps_remaining = _safe_int(state.get("lr_warmup_steps_remaining", 0), 0)
        manager._lr_warmup_factor = _safe_float(state.get("lr_warmup_factor", 1.0), 1.0)


        prev_stage_name = state.get("previous_stage", None)
        manager._previous_stage_name = prev_stage_name
        if prev_stage_name and manager._reward_blend_remaining > 0:
            try:
                manager._previous_stage_config = get_stage_config(CurriculumStage[prev_stage_name])
            except Exception as e:
                logger.warning(f"Could not restore previous_stage_config for blending: {e}; disabling blend.")
                manager._previous_stage_config = None
                manager._reward_blend_remaining = 0


        manager._last_episode_end_idx = _safe_int(state.get("last_episode_end_idx", -1), -1)


        processed_ids = state.get("processed_episode_ids", [])
        if processed_ids:
            trimmed = list(processed_ids)[-PROCESSED_EPISODE_ID_LIMIT:]
            manager._processed_episode_ids = deque(trimmed)
            manager._processed_episode_id_set = set(trimmed)


        manager._current_entropy = _safe_float(state.get("current_entropy", -1.0), -1.0)


        manager._review_tick_episode = _safe_int(state.get("review_tick_episode", -1), -1)


        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                manager._rng.bit_generator.state = rng_state
            except Exception as e:
                logger.warning(f"Could not restore RNG state: {e}")


        manager._validation_gate_history = state.get("validation_gate_history", []) or []

        validation_retry = state.get("validation_retry", {}) or {}
        retry_stage = validation_retry.get("stage")
        manager._last_validation_attempt_stage = (
            str(retry_stage) if retry_stage is not None else None
        )
        manager._last_validation_attempt_epoch = _safe_int(
            validation_retry.get("stage_epoch", -1), -1
        )
        manager._last_validation_attempt_stage_episode = _safe_int(
            validation_retry.get("stage_episode", -1), -1
        )


        manager._stress_test_history = state.get("stress_test_history", []) or []


        sel_state = state.get("selectivity_phase", {}) or {}
        if sel_state:
            manager._selectivity_phase_active = bool(sel_state.get("active", False))
            manager._selectivity_phase_episodes_remaining = _safe_int(
                sel_state.get("episodes_remaining", 0), 0
            )
            manager._selectivity_phase_requirements = sel_state.get("requirements", {}) or {}
            manager._selectivity_phase_completed_epoch = _safe_int(
                sel_state.get("completed_epoch", -1), -1
            )
            manager._selectivity_phase_failures = _safe_int(
                sel_state.get("failures", 0), 0
            )
            remaining = max(0, manager._selectivity_phase_episodes_remaining)
            failures = max(0, manager._selectivity_phase_failures)
            # Older checkpoints may not have target_episodes.  Count all
            # remaining episodes plus known failures as the minimum evidence;
            # unknown prior successes are deliberately not invented.
            fallback_target = remaining + failures
            manager._selectivity_phase_target_episodes = max(
                fallback_target,
                _safe_int(sel_state.get("target_episodes", fallback_target), fallback_target),
            )
            if manager._selectivity_phase_active and manager._selectivity_phase_target_episodes <= 0:
                manager._selectivity_phase_target_episodes = max(1, remaining)

        manager._rolling_stats_dirty = True
        return manager

    def load_from_dict(self, state: Dict[str, Any]) -> None:
        restored = self.__class__.from_dict(
            state,
            max_history_size=self.max_history_size,
            auto_promote=self.auto_promote,
            auto_demote=self.auto_demote,
            verbose=self.verbose,
            on_transition_callback=self.on_transition_callback,
            validation_evaluator=self.validation_evaluator,
            stress_test_evaluator=self.stress_test_evaluator,
            validation_gate_evaluator=self.validation_gate_evaluator,
            bars_per_trading_day=self.bars_per_trading_day,
        )
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        history_serialized: Dict[str, List[Dict[str, Any]]] = {}
        for stage, episodes in self._history.items():
            history_serialized[stage.name] = [asdict(ep) for ep in episodes]

        state = {
            "version": STATE_VERSION,
            "tz": self.tz,
            "current_stage": self.current_stage.name,
            "current_stage_epoch": self._current_stage_epoch,
            "total_timesteps": self.total_timesteps,
            "total_episodes": self.total_episodes,
            "stage_timesteps_total": {s.name: t for s, t in self._stage_timesteps_total.items()},
            "stage_episodes_total": {s.name: e for s, e in self._stage_episodes_total.items()},
            "stage_epoch_counter": {s.name: e for s, e in self._stage_epoch_counter.items()},
            "stage_timesteps_current": self.stage_timesteps,
            "stage_episodes_current": self.stage_episodes,
            "history": history_serialized,
            "transitions": self._transitions,
            "demotion_analyzer": self._demotion_analyzer.to_dict(),
            "recovery_state": self._recovery_state.to_dict(),
            "review_state": self._review_state.to_dict(),
            "learning_velocity": {
                "plateau_episodes": self._learning_velocity.plateau_episodes,
                "improvement_rates": self._learning_velocity.improvement_rates,
                "metric_history": {
                    k: list(v)[-100:] for k, v in self._learning_velocity.metric_history.items()
                } if hasattr(self._learning_velocity, 'metric_history') else {},
            },

            "transition_cooldown_remaining": self._transition_cooldown_remaining,
            "reward_blend_remaining": self._reward_blend_remaining,

            "previous_stage": self._previous_stage_name,
            "lr_warmup_active": self._lr_warmup_active,
            "lr_warmup_steps_remaining": self._lr_warmup_steps_remaining,
            "lr_warmup_factor": self._lr_warmup_factor,

            "last_episode_end_idx": self._last_episode_end_idx,

            "processed_episode_ids": list(self._processed_episode_ids),

            "current_entropy": self._current_entropy,

            "review_tick_episode": self._review_tick_episode,

            "rng_state": self._rng.bit_generator.state,

            "validation_gate_history": self._validation_gate_history[-100:],

            "validation_retry": {
                "stage": self._last_validation_attempt_stage,
                "stage_epoch": self._last_validation_attempt_epoch,
                "stage_episode": self._last_validation_attempt_stage_episode,
            },

            "stress_test_history": self._stress_test_history[-100:],

            "selectivity_phase": {
                "active": self._selectivity_phase_active,
                "episodes_remaining": self._selectivity_phase_episodes_remaining,
                "requirements": self._selectivity_phase_requirements,
                "completed_epoch": self._selectivity_phase_completed_epoch,
                "target_episodes": self._selectivity_phase_target_episodes,
                "failures": self._selectivity_phase_failures,
            },
            "saved_at": _now_iso(self.tz),
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, sort_keys=False)

        logger.info(f"Curriculum state saved to {path}")

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "CurriculumManager":
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)


        loaded_version = state.get("version", "1.0")
        if loaded_version != STATE_VERSION:
            logger.warning(
                f"Loading checkpoint from v{loaded_version} (current: v{STATE_VERSION}). "
                f"New features (recovery_state, review_state, learning_velocity) may use defaults."
            )

        tz = state.get("tz", DEFAULT_TZ)
        current_stage = CurriculumStage[state["current_stage"]]

        manager = cls(initial_stage=current_stage, tz=tz, **kwargs)
        manager.total_timesteps = _safe_int(state.get("total_timesteps", 0), 0)
        manager.total_episodes = _safe_int(state.get("total_episodes", 0), 0)


        for stage_name, timesteps in (state.get("stage_timesteps_total", {}) or {}).items():
            try:
                manager._stage_timesteps_total[CurriculumStage[stage_name]] = _safe_int(timesteps, 0)
            except KeyError:
                logger.debug(f"Unknown stage in timesteps_total: {stage_name}")

        for stage_name, episodes in (state.get("stage_episodes_total", {}) or {}).items():
            try:
                manager._stage_episodes_total[CurriculumStage[stage_name]] = _safe_int(episodes, 0)
            except KeyError:
                logger.debug(f"Unknown stage in episodes_total: {stage_name}")


        for stage_name, epoch in (state.get("stage_epoch_counter", {}) or {}).items():
            try:
                manager._stage_epoch_counter[CurriculumStage[stage_name]] = _safe_int(epoch, 0)
            except KeyError:
                logger.debug(f"Unknown stage in epoch_counter: {stage_name}")

        manager._current_stage_epoch = _safe_int(
            state.get("current_stage_epoch", manager._current_stage_epoch),
            manager._current_stage_epoch
        )
        manager.stage_timesteps = _safe_int(state.get("stage_timesteps_current", 0), 0)
        manager.stage_episodes = _safe_int(state.get("stage_episodes_current", 0), 0)


        progression = get_stage_progression()
        manager._history = {stage: deque(maxlen=manager.max_history_size) for stage in progression}
        for stage_name, episodes_data in (state.get("history", {}) or {}).items():
            try:
                stage = CurriculumStage[stage_name]
            except KeyError:
                logger.debug(f"Unknown stage in history: {stage_name}")
                continue
            for ep_data in (episodes_data or []):
                try:
                    manager._history[stage].append(EpisodeMetrics.from_dict(ep_data))
                except Exception as e:
                    logger.debug(f"Skipping malformed episode in history: {e}")
                    continue

        manager._transitions = state.get("transitions", []) or []


        demotion_data = state.get("demotion_analyzer", {})
        if demotion_data:
            manager._demotion_analyzer.load_from_dict(demotion_data)


        recovery_data = state.get("recovery_state", {})
        if recovery_data:
            manager._recovery_state = RecoveryProtocolState.from_dict(recovery_data)


        review_data = state.get("review_state", {})
        if review_data:
            manager._review_state = ReviewSessionState.from_dict(review_data)


        velocity_data = state.get("learning_velocity", {})
        if velocity_data:
            manager._learning_velocity.plateau_episodes = velocity_data.get("plateau_episodes", 0)
            manager._learning_velocity.improvement_rates = velocity_data.get("improvement_rates", {})

            metric_history = velocity_data.get("metric_history", {})
            if metric_history and hasattr(manager._learning_velocity, 'metric_history'):
                for k, v in metric_history.items():
                    manager._learning_velocity.metric_history[k] = deque(v, maxlen=100)


        manager._transition_cooldown_remaining = _safe_int(state.get("transition_cooldown_remaining", 0), 0)
        manager._reward_blend_remaining = _safe_int(state.get("reward_blend_remaining", 0), 0)
        manager._lr_warmup_active = bool(state.get("lr_warmup_active", False))
        manager._lr_warmup_steps_remaining = _safe_int(state.get("lr_warmup_steps_remaining", 0), 0)
        manager._lr_warmup_factor = _safe_float(state.get("lr_warmup_factor", 1.0), 1.0)


        prev_stage_name = state.get("previous_stage", None)
        manager._previous_stage_name = prev_stage_name
        if prev_stage_name and manager._reward_blend_remaining > 0:
            try:
                manager._previous_stage_config = get_stage_config(CurriculumStage[prev_stage_name])
            except Exception as e:
                logger.warning(f"Could not restore previous_stage_config for blending: {e}; disabling blend.")
                manager._previous_stage_config = None
                manager._reward_blend_remaining = 0


        manager._last_episode_end_idx = _safe_int(state.get("last_episode_end_idx", -1), -1)


        processed_ids = state.get("processed_episode_ids", [])
        if processed_ids:

            trimmed = list(processed_ids)[-PROCESSED_EPISODE_ID_LIMIT:]
            manager._processed_episode_ids = deque(trimmed)
            manager._processed_episode_id_set = set(trimmed)


        manager._current_entropy = _safe_float(state.get("current_entropy", -1.0), -1.0)


        manager._review_tick_episode = _safe_int(state.get("review_tick_episode", -1), -1)


        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                manager._rng.bit_generator.state = rng_state
            except Exception as e:
                logger.warning(f"Could not restore RNG state: {e}")


        manager._validation_gate_history = state.get("validation_gate_history", []) or []

        validation_retry = state.get("validation_retry", {}) or {}
        retry_stage = validation_retry.get("stage")
        manager._last_validation_attempt_stage = (
            str(retry_stage) if retry_stage is not None else None
        )
        manager._last_validation_attempt_epoch = _safe_int(
            validation_retry.get("stage_epoch", -1), -1
        )
        manager._last_validation_attempt_stage_episode = _safe_int(
            validation_retry.get("stage_episode", -1), -1
        )


        manager._stress_test_history = state.get("stress_test_history", []) or []


        sel_state = state.get("selectivity_phase", {}) or {}
        if sel_state:
            manager._selectivity_phase_active = bool(sel_state.get("active", False))
            manager._selectivity_phase_episodes_remaining = _safe_int(
                sel_state.get("episodes_remaining", 0), 0
            )
            manager._selectivity_phase_requirements = sel_state.get("requirements", {}) or {}
            manager._selectivity_phase_completed_epoch = _safe_int(
                sel_state.get("completed_epoch", -1), -1
            )
            manager._selectivity_phase_failures = _safe_int(
                sel_state.get("failures", 0), 0
            )
            remaining = max(0, manager._selectivity_phase_episodes_remaining)
            failures = max(0, manager._selectivity_phase_failures)
            fallback_target = remaining + failures
            manager._selectivity_phase_target_episodes = max(
                fallback_target,
                _safe_int(sel_state.get("target_episodes", fallback_target), fallback_target),
            )
            if manager._selectivity_phase_active and manager._selectivity_phase_target_episodes <= 0:
                manager._selectivity_phase_target_episodes = max(1, remaining)

        manager._rolling_stats_dirty = True

        logger.info(f"Curriculum state loaded from {path} (stage={current_stage.name}, epoch={manager._current_stage_epoch})")
        return manager

    def __repr__(self) -> str:
        return (
            f"CurriculumManager(stage={self.current_stage.name}, "
            f"epoch={self._current_stage_epoch}, "
            f"episodes={self.total_episodes}, "
            f"timesteps={self.total_timesteps:,})"
        )
