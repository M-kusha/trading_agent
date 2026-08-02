

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from envs.core.shared_utils import (
    clamp as _clamp,
)
from envs.core.shared_utils import (
    get_envs_logger,
    iso_timestamp,
)
from envs.core.shared_utils import (
    safe_float as _sf,
)
from envs.core.shared_utils import (
    safe_int as _si,
)
from envs.curriculum.config.protocols import RecoveryProtocolConfig
from envs.curriculum.config.stages import (
    CurriculumStage,
    TradingSkill,
)
from envs.curriculum.config.thresholds import SkillRequirements

if TYPE_CHECKING:
    from envs.curriculum.metrics import EpisodeMetrics, RollingStats
    from envs.curriculum.protocols import RecoveryProtocolState

logger = get_envs_logger("curriculum.skills")

DEFAULT_TZ = "Europe/Berlin"
DEMOTION_HISTORY_LIMIT = 200


def _is_finite(x: Any) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _finite_floats(xs: List[Any]) -> List[float]:
    out: List[float] = []
    for v in xs:
        fv = _sf(v, None)  # type: ignore[arg-type]
        if fv is not None and _is_finite(fv):
            out.append(float(fv))
    return out


def _now_iso(tz: str = DEFAULT_TZ) -> str:
    try:
        return datetime.now(tz=ZoneInfo(tz)).isoformat()
    except Exception as e:
        logger.debug(f"Timezone fallback for {tz}: {e}")
        return iso_timestamp()


def _foundation_stage() -> CurriculumStage:


    return sorted(CurriculumStage, key=lambda s: s.value)[0]


def _safe_float(x: Any, default: float = 0.0) -> float:
    return _sf(x, default)


def _safe_int(x: Any, default: int = 0) -> int:
    return _si(x, default)


@dataclass
class SkillAssessment:
    skill_scores: Dict[TradingSkill, float] = field(default_factory=dict)
    skill_confidence: Dict[TradingSkill, float] = field(default_factory=dict)


    weighted_average: float = 0.5
    weakest_skills: List[TradingSkill] = field(default_factory=list)
    strongest_skills: List[TradingSkill] = field(default_factory=list)

    @classmethod
    def from_episode_results(
        cls,
        episodes: List["EpisodeMetrics"],
        stats: "RollingStats",
        bars_per_trading_day: int = 96,
        regime_skill_vector: Optional[Dict[str, float]] = None,
    ) -> "SkillAssessment":
        scores: Dict[TradingSkill, float] = {}
        confidence: Dict[TradingSkill, float] = {}

        n = len(episodes)
        if n == 0:
            return cls()

        base_conf = min(1.0, n / 50)


        entry_qualities = _finite_floats(
            [e.avg_entry_quality for e in episodes if getattr(e, "trade_count", 0) > 0]
        )
        if entry_qualities:
            eq = float(np.mean(entry_qualities))
            scores[TradingSkill.ENTRY_TIMING] = float(_clamp(eq, 0.0, 1.0))
            confidence[TradingSkill.ENTRY_TIMING] = base_conf


        total_exits = sum(
            e.trailing_stop_exits + e.agent_close_exits + e.hard_stop_exits +
            e.risk_liquidation_exits + e.other_exits
            for e in episodes
        )
        good_exits = sum(e.trailing_stop_exits + e.agent_close_exits for e in episodes)
        if total_exits > 0:
            eq = good_exits / total_exits
            scores[TradingSkill.EXIT_QUALITY] = float(_clamp(eq, 0.0, 1.0))
            confidence[TradingSkill.EXIT_QUALITY] = min(1.0, total_exits / 50)


        dd_rate = float(_sf(getattr(stats, "dd_breach_rate", 0.0), 0.0))
        scores[TradingSkill.DRAWDOWN_CONTROL] = float(_clamp(1.0 - dd_rate, 0.0, 1.0))
        confidence[TradingSkill.DRAWDOWN_CONTROL] = base_conf


        bars_between = float(_sf(getattr(stats, "mean_bars_between_trades", 0.0), 0.0))
        if bars_between > 0:

            target_bars_between = max(1.0, bars_per_trading_day / 3.0)
            patience_score = _clamp(bars_between / target_bars_between, 0.0, 1.0)
        else:

            avg_trades_per_episode = stats.mean_trade_count
            episode_lengths = _finite_floats([e.episode_length for e in episodes if getattr(e, "episode_length", 0) > 0])
            if episode_lengths:
                avg_episode_bars = float(np.mean(episode_lengths))
            else:
                avg_episode_bars = 2000.0
            est_trading_days_per_episode = max(1.0, avg_episode_bars / bars_per_trading_day)
            trades_per_day = avg_trades_per_episode / est_trading_days_per_episode
            if trades_per_day <= 0.5:
                patience_score = 1.0
            elif trades_per_day <= 1.0:
                patience_score = 0.90
            elif trades_per_day <= 1.5:
                patience_score = 0.75
            elif trades_per_day <= 2.0:
                patience_score = 0.60
            elif trades_per_day <= 3.0:
                patience_score = 0.40
            elif trades_per_day <= 4.0:
                patience_score = 0.20
            else:
                patience_score = max(0.0, 0.20 - (trades_per_day - 4.0) / 5.0)
        scores[TradingSkill.PATIENCE] = float(patience_score)
        confidence[TradingSkill.PATIENCE] = base_conf


        skipped = float(_sf(getattr(stats, "mean_setup_skipped_per_episode", 0.0), 0.0))
        trades = float(_sf(getattr(stats, "mean_trade_count", 0.0), 0.0))
        total_setups = max(skipped + trades, 0.0)
        if total_setups > 0:
            skipped_rate = skipped / total_setups
            target_skipped_rate = 0.50
            selectivity_score = 1.0 - abs(skipped_rate - target_skipped_rate) / target_skipped_rate
        else:
            selectivity_score = 0.5
        scores[TradingSkill.SELECTIVITY] = float(_clamp(selectivity_score, 0.0, 1.0))
        confidence[TradingSkill.SELECTIVITY] = base_conf


        certainty = float(_sf(getattr(stats, "mean_entry_certainty", 0.0), 0.0))
        scores[TradingSkill.CERTAINTY] = float(_clamp(certainty, 0.0, 1.0))
        confidence[TradingSkill.CERTAINTY] = base_conf


        setup_quality = float(_sf(getattr(stats, "mean_setup_quality", 0.0), 0.0))
        scores[TradingSkill.SETUP_QUALITY] = float(_clamp(setup_quality, 0.0, 1.0))
        confidence[TradingSkill.SETUP_QUALITY] = base_conf


        fomo_rate = float(_sf(getattr(stats, "fomo_trade_rate", 0.0), 0.0))
        revenge_rate = float(_sf(getattr(stats, "revenge_trade_rate", 0.0), 0.0))
        discipline_score = 1.0 - min(1.0, (fomo_rate + revenge_rate) / 2.0)
        scores[TradingSkill.DISCIPLINE] = float(_clamp(discipline_score, 0.0, 1.0))
        confidence[TradingSkill.DISCIPLINE] = base_conf


        r_mult = stats.mean_r_multiple

        r_score = _clamp((r_mult + 0.5) / 1.0, 0.0, 1.0)
        scores[TradingSkill.RISK_REWARD] = r_score
        confidence[TradingSkill.RISK_REWARD] = base_conf


        wr_std = stats.std_win_rate

        cons_score = _clamp(1.0 - wr_std / 0.3, 0.0, 1.0)
        scores[TradingSkill.CONSISTENCY] = cons_score
        confidence[TradingSkill.CONSISTENCY] = base_conf


        cl_breach = float(_sf(getattr(stats, "consecutive_loss_breach_rate", 0.0), 0.0))
        scores[TradingSkill.LOSS_MANAGEMENT] = float(_clamp(1.0 - cl_breach, 0.0, 1.0))
        confidence[TradingSkill.LOSS_MANAGEMENT] = base_conf


        wr = float(_sf(getattr(stats, "mean_win_rate", 0.0), 0.0))
        trend_proxy = float(_clamp(wr * 1.5, 0.0, 1.0))
        scores[TradingSkill.TREND_ALIGNMENT] = trend_proxy
        confidence[TradingSkill.TREND_ALIGNMENT] = base_conf


        pf = float(_sf(getattr(stats, "mean_profit_factor", 0.0), 0.0))
        adapt_proxy = float(_clamp(pf / 2.0, 0.0, 1.0))
        scores[TradingSkill.ADAPTATION] = adapt_proxy
        confidence[TradingSkill.ADAPTATION] = base_conf * 0.7


        if stats.mean_drawdown > 0 and stats.mean_trade_count > 0:
            dd_per_trade = stats.mean_drawdown / stats.mean_trade_count

            sizing_score = _clamp(1.0 - dd_per_trade * 10, 0.0, 1.0)
        else:
            sizing_score = 0.5
        scores[TradingSkill.POSITION_SIZING] = sizing_score
        confidence[TradingSkill.POSITION_SIZING] = base_conf * 0.8


        if regime_skill_vector:

            if "adaptation" in regime_skill_vector:
                scores[TradingSkill.ADAPTATION] = float(_clamp(_sf(regime_skill_vector.get("adaptation"), adapt_proxy), 0.0, 1.0))
                confidence[TradingSkill.ADAPTATION] = base_conf
            if "trend_following" in regime_skill_vector:
                scores[TradingSkill.TREND_ALIGNMENT] = float(_clamp(_sf(regime_skill_vector.get("trend_following"), trend_proxy), 0.0, 1.0))
                confidence[TradingSkill.TREND_ALIGNMENT] = base_conf


        total_weight = sum(confidence.values())
        if total_weight > 0:
            weighted_avg = sum(
                scores[skill] * confidence[skill]
                for skill in scores
            ) / total_weight
        else:
            weighted_avg = 0.5


        weighted_scores = {
            skill: scores[skill] * confidence.get(skill, 0.5)
            for skill in scores
        }
        sorted_skills = sorted(weighted_scores.items(), key=lambda x: x[1])
        weakest = [s[0] for s in sorted_skills[:3]]
        strongest = [s[0] for s in sorted_skills[-3:][::-1]]

        return cls(
            skill_scores=scores,
            skill_confidence=confidence,
            weighted_average=weighted_avg,
            weakest_skills=weakest,
            strongest_skills=strongest,
        )

    def check_requirements(
        self,
        requirements: SkillRequirements,
    ) -> Tuple[bool, Dict[str, Any]]:
        results: Dict[str, Any] = {"checks": {}, "passed_all": True}
        passed_count = 0
        total_required = len(requirements.required_skills)

        for skill, min_score in requirements.required_skills.items():
            actual = self.skill_scores.get(skill, 0.0)
            conf = self.skill_confidence.get(skill, 0.0)


            passed = (actual >= min_score) and (conf >= requirements.min_confidence)

            results["checks"][skill.value] = {
                "required": min_score,
                "actual": actual,
                "confidence": conf,
                "passed": passed,
            }

            if passed:
                passed_count += 1
            else:
                results["passed_all"] = False


        if not requirements.require_all_skills:

            numer = 0.0
            denom = 0.0
            for skill, min_score in requirements.required_skills.items():
                s = float(self.skill_scores.get(skill, 0.0))
                c = float(self.skill_confidence.get(skill, 0.0))
                w = float(requirements.get_weight(skill))

                if c >= requirements.min_confidence:
                    numer += s * w
                    denom += w

            weighted = (numer / denom) if denom > 0 else 0.0
            meets_weighted = weighted >= requirements.weighted_threshold
            results["weighted_average"] = weighted
            results["weighted_threshold"] = requirements.weighted_threshold
            results["meets_weighted"] = meets_weighted
            results["overall_passed"] = meets_weighted or results["passed_all"]
        else:
            results["overall_passed"] = results["passed_all"]

        results["passed_count"] = passed_count
        results["total_required"] = total_required

        return results["overall_passed"], results

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_scores": {k.value: v for k, v in self.skill_scores.items()},
            "skill_confidence": {k.value: v for k, v in self.skill_confidence.items()},
            "weighted_average": self.weighted_average,
            "weakest_skills": [s.value for s in self.weakest_skills],
            "strongest_skills": [s.value for s in self.strongest_skills],
        }


@dataclass
class DemotionRecord:
    from_stage: CurriculumStage
    to_stage: CurriculumStage
    timestamp: str
    failure_reasons: List[str]
    skill_assessment: Optional[Dict[str, Any]]
    stats_snapshot: Optional[Dict[str, Any]]
    global_episode: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_stage": self.from_stage.name,
            "to_stage": self.to_stage.name,
            "timestamp": self.timestamp,
            "failure_reasons": self.failure_reasons,
            "skill_assessment": self.skill_assessment,
            "stats_snapshot": self.stats_snapshot,
            "global_episode": self.global_episode,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DemotionRecord":
        from_stage = _foundation_stage()
        to_stage = _foundation_stage()
        try:
            if d.get("from_stage"):
                from_stage = CurriculumStage[d["from_stage"]]
            if d.get("to_stage"):
                to_stage = CurriculumStage[d["to_stage"]]
        except Exception as e:
            logger.debug(f"Could not parse stage names from demotion record: {e}")
        return cls(
            from_stage=from_stage,
            to_stage=to_stage,
            timestamp=str(d.get("timestamp", "")),
            failure_reasons=list(d.get("failure_reasons", []) or []),
            skill_assessment=d.get("skill_assessment", None),
            stats_snapshot=d.get("stats_snapshot", None),
            global_episode=_safe_int(d.get("global_episode", 0), 0),
        )


class DemotionAnalyzer:

    def __init__(self) -> None:
        self.demotion_history: List[DemotionRecord] = []
        self.stage_failure_counts: Dict[CurriculumStage, int] = {}

    def record_demotion(
        self,
        from_stage: CurriculumStage,
        to_stage: CurriculumStage,
        failure_reasons: List[str],
        skill_assessment: Optional[SkillAssessment],
        stats: Optional["RollingStats"],
        global_episode: int,
    ) -> None:
        record = DemotionRecord(
            from_stage=from_stage,
            to_stage=to_stage,
            timestamp=_now_iso(),
            failure_reasons=failure_reasons,
            skill_assessment=skill_assessment.to_dict() if skill_assessment else None,
            stats_snapshot=stats.to_dict() if stats else None,
            global_episode=global_episode,
        )
        self.demotion_history.append(record)
        self.stage_failure_counts[from_stage] = self.stage_failure_counts.get(from_stage, 0) + 1


        if len(self.demotion_history) > DEMOTION_HISTORY_LIMIT:
            self.demotion_history = self.demotion_history[-DEMOTION_HISTORY_LIMIT:]

    def get_failure_count(self, stage: CurriculumStage) -> int:
        return self.stage_failure_counts.get(stage, 0)

    def diagnose_repeated_failures(self, stage: CurriculumStage) -> Dict[str, Any]:
        relevant = [d for d in self.demotion_history if d.from_stage == stage]

        if len(relevant) < 2:
            return {"status": "insufficient_data", "failure_count": len(relevant)}


        reason_counts: Dict[str, int] = {}
        for d in relevant:
            for reason in d.failure_reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1


        skill_scores_all: Dict[str, List[float]] = {}
        for d in relevant:
            if d.skill_assessment:
                for skill_name, score in d.skill_assessment.get("skill_scores", {}).items():
                    if skill_name not in skill_scores_all:
                        skill_scores_all[skill_name] = []
                    skill_scores_all[skill_name].append(score)

        avg_skills: Dict[str, float] = {
            skill: float(np.mean(scores))
            for skill, scores in skill_scores_all.items()
        }


        recommendation = self._generate_recommendation(reason_counts, avg_skills)

        return {
            "status": "analyzed",
            "failure_count": len(relevant),
            "top_failure_reasons": sorted(reason_counts.items(), key=lambda x: -x[1])[:3],
            "weak_skills": sorted(avg_skills.items(), key=lambda x: x[1])[:3],
            "recommendation": recommendation,
        }

    def _generate_recommendation(
        self,
        reasons: Dict[str, int],
        skills: Dict[str, float],
    ) -> Dict[str, Any]:
        focus_skill: Optional[TradingSkill] = None
        reward_mods: Dict[str, float] = {}
        constraint_mods: Dict[str, float] = {}
        description = "general_practice"


        if reasons.get("drawdown_critical", 0) + reasons.get("dd_breach_critical", 0) > 1:
            focus_skill = TradingSkill.DRAWDOWN_CONTROL
            reward_mods = {"dd_penalty_scale": 2.0, "reward_scale": 0.8}
            constraint_mods = {"max_trades_per_day": 0.7}
            description = "focus_drawdown_control"

        elif (
            reasons.get("excessive_fomo_trading", 0)
            + reasons.get("revenge_trading", 0)
            + reasons.get("overtrading", 0)
        ) > 0:

            try:
                from envs.curriculum.protocols import RecoveryProtocolState
                rec = RecoveryProtocolState.get_patience_focused_recovery()
            except Exception:
                rec = {
                    "reward_modifications": {"churn_penalty_per_trade": 1.5},
                    "constraint_modifications": {"max_trades_per_day": 0.5},
                    "description": "focus_patience_discipline",
                }
            focus_skill = TradingSkill.DISCIPLINE
            reward_mods = rec.get("reward_modifications", {})
            constraint_mods = rec.get("constraint_modifications", {})
            description = rec.get("description", "focus_patience_discipline")

        elif reasons.get("win_rate_critical", 0) > 1:
            focus_skill = TradingSkill.ENTRY_TIMING
            reward_mods = {"entry_quality_weight": 0.5, "soft_block_penalty": 0.1}
            description = "focus_entry_quality"

        elif reasons.get("profit_factor_critical", 0) > 1:
            focus_skill = TradingSkill.RISK_REWARD
            reward_mods = {"r_multiple_bonus_scale": 0.5, "time_efficiency_scale": 0.25}
            description = "focus_risk_reward"

        elif skills.get(TradingSkill.DISCIPLINE.value, 1.0) < 0.4:
            focus_skill = TradingSkill.DISCIPLINE
            reward_mods = {"loss_streak_caution_base": 0.04}
            constraint_mods = {"daily_trade_soft_limit": 0.5}
            description = "focus_discipline"

        elif skills.get(TradingSkill.SELECTIVITY.value, 1.0) < 0.4:
            focus_skill = TradingSkill.SELECTIVITY
            reward_mods = {"churn_penalty_per_trade": 0.05}
            constraint_mods = {"daily_trade_soft_limit": 0.6}
            description = "increase_selectivity"

        elif skills.get(TradingSkill.PATIENCE.value, 1.0) < 0.4:
            focus_skill = TradingSkill.PATIENCE
            reward_mods = {"churn_penalty_per_trade": 0.05}
            constraint_mods = {"daily_trade_soft_limit": 0.5}
            description = "reduce_trade_frequency"

        elif skills.get(TradingSkill.LOSS_MANAGEMENT.value, 1.0) < 0.4:
            focus_skill = TradingSkill.LOSS_MANAGEMENT
            reward_mods = {"loss_streak_penalty_per_loss": 0.05}
            description = "focus_loss_management"

        return {
            "focus_skill": focus_skill.value if focus_skill else None,
            "reward_modifications": reward_mods,
            "constraint_modifications": constraint_mods,
            "description": description,
        }

    def create_recovery_protocol(
        self,
        stage: CurriculumStage,
        config: RecoveryProtocolConfig,
    ) -> "RecoveryProtocolState":

        from envs.curriculum.protocols import RecoveryProtocolState

        if not config.enabled:
            return RecoveryProtocolState()

        failure_count = self.get_failure_count(stage)
        if failure_count < config.trigger_after_demotions:
            return RecoveryProtocolState()

        diagnosis = self.diagnose_repeated_failures(stage)
        if diagnosis.get("status") != "analyzed":
            return RecoveryProtocolState()

        rec = diagnosis.get("recommendation", {})

        focus_skill = None
        if rec.get("focus_skill"):
            try:

                focus_skill = TradingSkill(rec["focus_skill"])
            except Exception:

                try:
                    focus_skill = TradingSkill[rec["focus_skill"]]
                except Exception:
                    logger.debug(f"Unknown focus_skill in recommendation: {rec.get('focus_skill')}")


        if config.focus_skill:
            focus_skill = config.focus_skill


        reward_mods = {**rec.get("reward_modifications", {}), **config.reward_modifications}
        constraint_mods = {**rec.get("constraint_modifications", {}), **config.constraint_modifications}

        return RecoveryProtocolState(
            triggered=True,
            focus_skill=focus_skill,
            reward_modifications=reward_mods,
            constraint_modifications=constraint_mods,
            episodes_remaining=config.recovery_duration_episodes,
            trigger_reason=rec.get("description", "repeated_failures"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "demotion_count": len(self.demotion_history),
            "stage_failure_counts": {s.name: c for s, c in self.stage_failure_counts.items()},
            "recent_demotions": [d.to_dict() for d in self.demotion_history[-5:]],

            "demotion_history": [d.to_dict() for d in self.demotion_history[-DEMOTION_HISTORY_LIMIT:]],
        }

    def load_from_dict(self, data: Dict[str, Any]) -> None:
        self.stage_failure_counts = {}
        for stage_name, count in data.get("stage_failure_counts", {}).items():
            try:
                stage = CurriculumStage[stage_name]
                self.stage_failure_counts[stage] = count
            except KeyError:
                logger.debug(f"Unknown stage name in failure counts: {stage_name}")


        self.demotion_history = []
        history_data = data.get("demotion_history") or data.get("recent_demotions") or []
        for d in history_data[-DEMOTION_HISTORY_LIMIT:]:
            try:
                self.demotion_history.append(DemotionRecord.from_dict(d))
            except Exception as e:
                logger.debug(f"Skipping malformed demotion record: {e}")


__all__ = [
    "DemotionAnalyzer",
    "DemotionRecord",
    "SkillAssessment",
]
