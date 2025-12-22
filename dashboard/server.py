#!/usr/bin/env python3
"""
PropFirm PPO Training Dashboard Server v2.1
===========================================

Fixes vs v2.0:
- Normalizes promotion gate keys (e.g. "insufficient data" -> "data_sufficiency")
- Computes strict promotion readiness to prevent UI contradictions:
    strict_ready = composite_score.promotion_ready AND all prerequisites passed
- Normalizes stage_history entries so UI doesn't show "Stage —"
- Exposes requirements_by_stage from envs.curriculum_config (optional source of truth)
- Adds GET /api/requirements endpoint
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
import argparse
import logging

# Ensure project root is importable (dashboard/ is typically one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Web framework - conditional import
WEB_AVAILABLE = False
if TYPE_CHECKING:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    WEB_AVAILABLE = True
except ImportError:
    print("[Dashboard] FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn websockets")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    metrics_file: str = "logs/training/live_metrics.json"
    update_interval: float = 0.5
    file_poll_interval: float = 0.1
    max_history_points: int = 500


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS FOR STATUS COLORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricThresholds:
    win_rate_good: float = 55.0
    win_rate_ok: float = 45.0
    drawdown_good: float = 3.0
    drawdown_ok: float = 6.0
    ev_good: float = 0.5
    ev_ok: float = 0.2
    entropy_good_min: float = -9.0
    entropy_good_max: float = -5.0
    kl_good: float = 0.015
    kl_ok: float = 0.025
    clip_good_min: float = 0.05
    clip_good_max: float = 0.20
    r_mult_good: float = 0.5
    r_mult_ok: float = 0.0
    pf_good: float = 1.5
    pf_ok: float = 1.0
    eq_good: float = 0.6
    eq_ok: float = 0.45


THRESHOLDS = MetricThresholds()


def get_status_color(value: float, good_threshold: float, ok_threshold: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= good_threshold:
            return "good"
        if value >= ok_threshold:
            return "ok"
        return "bad"
    else:
        if value <= good_threshold:
            return "good"
        if value <= ok_threshold:
            return "ok"
        return "bad"


def get_range_status(value: float, good_min: float, good_max: float) -> str:
    if good_min <= value <= good_max:
        return "good"
    if abs(value - (good_min + good_max) / 2) < abs(good_max - good_min):
        return "ok"
    return "bad"


# ═══════════════════════════════════════════════════════════════════════════════
# CURRICULUM REQUIREMENTS EXPORT (optional source of truth)
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_asdict(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_asdict(v) for k, v in obj.items()}
    try:
        # is_dataclass returns True for both types and instances, but asdict only works on instances
        if is_dataclass(obj) and not isinstance(obj, type):
            return _safe_asdict(asdict(obj))
    except Exception:
        pass
    if hasattr(obj, "__dict__"):
        try:
            return {str(k): _safe_asdict(v) for k, v in obj.__dict__.items() if not str(k).startswith("_")}
        except Exception:
            return str(obj)
    return str(obj)


def _extract_stage_requirements(stage_cfg: Any) -> Dict[str, Any]:
    cfg_dict = _safe_asdict(stage_cfg) if stage_cfg is not None else {}
    competence = {}
    if isinstance(cfg_dict, dict) and isinstance(cfg_dict.get("competence"), dict):
        competence = cfg_dict.get("competence", {})  # type: ignore

    skill_requirements = {}
    if isinstance(cfg_dict, dict):
        skill_requirements = cfg_dict.get("skill_requirements") or cfg_dict.get("skills") or cfg_dict.get("skill_thresholds") or {}

    entropy_targets = {}
    if isinstance(cfg_dict, dict):
        entropy_targets = cfg_dict.get("entropy_targets") or cfg_dict.get("entropy") or {}

    composite = {}
    if isinstance(cfg_dict, dict):
        composite = cfg_dict.get("composite_scoring") or cfg_dict.get("composite") or cfg_dict.get("composite_score") or {}

    # Flatten easy numeric thresholds
    promotion_criteria: Dict[str, Any] = {}
    if isinstance(competence, dict):
        for k, v in competence.items():
            if isinstance(v, (int, float, bool, str)) or v is None:
                promotion_criteria[str(k)] = v

    return {
        "competence": competence or {},
        "skill_requirements": skill_requirements or {},
        "entropy_targets": entropy_targets or {},
        "composite": composite or {},
        "promotion_criteria": promotion_criteria,
        "raw": cfg_dict if isinstance(cfg_dict, dict) else {"value": cfg_dict},
    }


@lru_cache(maxsize=1)
def get_requirements_by_stage() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "available": False,
        "generated_at": datetime.now().isoformat(),
        "source": None,
        "stages": {},
        "error": None,
    }

    try:
        from envs.curriculum_config import CurriculumStage, get_stage_config  # type: ignore

        stages_out: Dict[str, Any] = {}
        for stage in list(CurriculumStage):
            stage_name = getattr(stage, "name", str(stage))
            stage_idx = getattr(stage, "value", None)

            cfg = None
            try:
                cfg = get_stage_config(stage)
            except Exception:
                if stage_idx is not None:
                    cfg = get_stage_config(stage_idx)

            stages_out[stage_name] = {
                "stage_index": stage_idx,
                "requirements": _extract_stage_requirements(cfg),
            }

        payload["available"] = True
        payload["source"] = "envs.curriculum_config.get_stage_config"
        payload["stages"] = stages_out
        return payload

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        return payload


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS READER
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsReader:
    def __init__(self, metrics_file: str, max_history: int = 500):
        self.metrics_file = Path(metrics_file)
        self.max_history = max_history
        self._last_modified: float = 0.0
        self._last_data: Dict[str, Any] = {}
        self._history: Dict[str, List[float]] = {
            "rewards": [],
            "pnls": [],
            "win_rates": [],
            "drawdowns": [],
            "entropy": [],
            "explained_variance": [],
            "kl_divergence": [],
            "r_multiples": [],
            "trades_per_episode": [],
        }
        self._lock = threading.Lock()

    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            if f != f:
                return default
            return f
        except (TypeError, ValueError):
            return default

    def _safe_int(self, val: Any, default: int = 0) -> int:
        try:
            return int(self._safe_float(val, float(default)))
        except Exception:
            return default

    def _safe_list(self, val: Any, default: Optional[List] = None) -> List:
        if default is None:
            default = []
        return val if isinstance(val, list) else default

    def _safe_dict(self, val: Any, default: Optional[Dict] = None) -> Dict:
        if default is None:
            default = {}
        return val if isinstance(val, dict) else default

    def _safe_bool(self, val: Any, default: bool = False) -> bool:
        return val if isinstance(val, bool) else default

    def _append_history(self, key: str, value: float) -> None:
        if key in self._history:
            self._history[key].append(value)
            if len(self._history[key]) > self.max_history:
                self._history[key] = self._history[key][-self.max_history:]

    def read_metrics(self) -> Dict[str, Any]:
        with self._lock:
            if not self.metrics_file.exists():
                return self._get_empty_metrics("Waiting for training to start...")

            try:
                mtime = self.metrics_file.stat().st_mtime
                if mtime == self._last_modified and self._last_data:
                    return self._last_data

                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                self._last_modified = mtime
                processed = self._process_metrics(raw)
                self._last_data = processed
                return processed

            except json.JSONDecodeError as e:
                if self._last_data:
                    return self._last_data
                logger.debug(f"JSON decode error: {e}")
                return self._get_empty_metrics("Reading metrics...")
            except Exception as e:
                logger.warning(f"Error reading metrics: {e}")
                return self._last_data if self._last_data else self._get_empty_metrics(str(e))

    def _get_empty_metrics(self, message: str = "") -> Dict[str, Any]:
        return {
            "status": "waiting",
            "message": message,
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "progress": {"timesteps": 0, "total_timesteps": 0, "progress_pct": 0, "total_episodes": 0},
            "learning": {},
            "trading": {},
            "quality": {},
            "exit_stats": {"distribution": {}},
            "curriculum_stage": "N/A",
            "curriculum_stage_idx": 0,
            "curriculum_progress": {},
            "curriculum_detail": {},
            "recent_rewards": [],
            "recent_pnls": [],
            "recent_win_rates": [],
            "recent_drawdowns": [],
            "stage_history": [],
            "requirements_by_stage": get_requirements_by_stage(),
        }

    def _normalize_promotion_key(self, k: str) -> str:
        k0 = (k or "").strip()
        if not k0:
            return k0
        k1 = k0.lower().strip().replace(" ", "_").replace("-", "_")

        # Canonicalize common gates
        if k1 in ("min_episodes", "episodes", "stage_episodes"):
            return "min_episodes"
        if k1 in ("min_timesteps", "timesteps", "stage_timesteps"):
            return "min_timesteps"
        if k1 in ("insufficient_data", "sufficient_data", "data_sufficiency", "data_sufficient", "insufficientdata"):
            return "data_sufficiency"

        # Keep as-is (snake-ish)
        return k1

    def _normalize_promotion_checks(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (checks or {}).items():
            nk = self._normalize_promotion_key(str(k))
            if not isinstance(v, dict):
                continue
            out[nk] = {
                "required": v.get("required"),
                "actual": v.get("actual"),
                "passed": bool(v.get("passed", False)),
                "original_key": k,
            }
        return out

    def _normalize_stage_history(self, hist: Any) -> List[Dict[str, Any]]:
        if not isinstance(hist, list):
            return []
        out: List[Dict[str, Any]] = []
        for e in hist:
            if not isinstance(e, dict):
                continue
            stage_val = e.get("stage")
            stage_name = e.get("stage_name") or e.get("stage_label") or e.get("stage_str")
            stage_index = e.get("stage_index")
            if stage_val is None and stage_index is None and stage_name is None:
                # Sometimes called "from_stage"/"to_stage"
                stage_val = e.get("to_stage") or e.get("stage_to") or e.get("current_stage")

            out.append({
                "stage": stage_val,
                "stage_index": stage_index,
                "stage_name": stage_name,
                "timestep": e.get("timestep") or e.get("timesteps") or e.get("step"),
                "episode": e.get("episode") or e.get("ep"),
                "reason": e.get("reason") or e.get("message") or "",
                "raw": e,
            })
        return out

    def _process_metrics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        # ─────────────────────────────────────────────────────────────
        # PROGRESS
        # ─────────────────────────────────────────────────────────────
        progress_section = raw.get("progress", {})
        timesteps = self._safe_int(progress_section.get("timesteps", raw.get("timesteps", 0)))
        total_timesteps = self._safe_int(progress_section.get("total_timesteps", raw.get("total_timesteps", 1)))
        progress_pct = self._safe_float(progress_section.get("progress_pct", raw.get("progress_pct", 0)))
        total_episodes = self._safe_int(progress_section.get("total_episodes", raw.get("total_episodes", 0)))

        progress = {
            "timesteps": timesteps,
            "total_timesteps": total_timesteps,
            "progress_pct": progress_pct,
            "total_episodes": total_episodes,
            "eta_seconds": self._estimate_eta(timesteps, total_timesteps, raw),
        }

        # ─────────────────────────────────────────────────────────────
        # LEARNING METRICS (PPO)
        # ─────────────────────────────────────────────────────────────
        learning_section = raw.get("learning", {})
        mean_reward = self._safe_float(learning_section.get("mean_reward", raw.get("mean_reward", 0)))
        total_pnl = self._safe_float(learning_section.get("total_pnl", raw.get("total_pnl", 0)))

        approx_kl = self._safe_float(learning_section.get("kl_divergence", learning_section.get("approx_kl", raw.get("approx_kl", 0))))
        clip_fraction = self._safe_float(learning_section.get("clip_fraction", raw.get("clip_fraction", 0)))
        entropy = self._safe_float(learning_section.get("entropy", raw.get("entropy", 0)))
        explained_variance = self._safe_float(learning_section.get("explained_variance", raw.get("explained_variance", 0)))
        value_loss = self._safe_float(learning_section.get("value_loss", raw.get("value_loss", 0)))
        policy_loss = self._safe_float(learning_section.get("policy_loss", raw.get("policy_loss", 0)))
        learning_rate = self._safe_float(learning_section.get("learning_rate", raw.get("learning_rate", 3e-4)))
        fps = self._safe_float(learning_section.get("fps", raw.get("fps", 0)))
        n_updates = self._safe_int(learning_section.get("n_updates", raw.get("n_updates", 0)))
        clip_range = self._safe_float(learning_section.get("clip_range", raw.get("clip_range", 0.2)))

        self._append_history("entropy", entropy)
        self._append_history("explained_variance", explained_variance)
        self._append_history("kl_divergence", approx_kl)

        learning = {
            "mean_reward": mean_reward,
            "mean_reward_status": get_status_color(mean_reward, 0.5, 0, True),
            "total_pnl": total_pnl,
            "total_pnl_status": get_status_color(total_pnl, 500, 0, True),
            "approx_kl": approx_kl,
            "approx_kl_status": get_status_color(approx_kl, THRESHOLDS.kl_good, THRESHOLDS.kl_ok, False),
            "clip_fraction": clip_fraction,
            "clip_fraction_status": get_range_status(clip_fraction, THRESHOLDS.clip_good_min, THRESHOLDS.clip_good_max),
            "entropy": entropy,
            "entropy_status": get_range_status(entropy, THRESHOLDS.entropy_good_min, THRESHOLDS.entropy_good_max),
            "explained_variance": explained_variance,
            "explained_variance_status": get_status_color(explained_variance, THRESHOLDS.ev_good, THRESHOLDS.ev_ok, True),
            "value_loss": value_loss,
            "value_loss_status": get_status_color(value_loss, 0.1, 0.5, False),
            "policy_loss": policy_loss,
            "policy_loss_status": "good" if policy_loss < 0 else ("ok" if policy_loss < 0.05 else "bad"),
            "learning_rate": learning_rate,
            "fps": fps,
            "fps_status": get_status_color(fps, 100, 50, True),
            "n_updates": n_updates,
            "clip_range": clip_range,
        }

        # ─────────────────────────────────────────────────────────────
        # TRADING METRICS
        # ─────────────────────────────────────────────────────────────
        trading_section = raw.get("trading", {})
        mean_win_rate_raw = trading_section.get("mean_win_rate", raw.get("mean_win_rate", 0))
        mean_win_rate = self._safe_float(mean_win_rate_raw)
        if 0 < mean_win_rate < 1:
            mean_win_rate *= 100

        max_drawdown_raw = trading_section.get("max_drawdown", raw.get("max_drawdown", 0))
        max_drawdown = self._safe_float(max_drawdown_raw)
        if 0 < max_drawdown < 1:
            max_drawdown *= 100

        mean_trades = self._safe_float(trading_section.get("mean_trades", raw.get("mean_trades", 0)))
        total_trades = self._safe_int(trading_section.get("total_trades", raw.get("total_trades", 0)))

        self._append_history("win_rates", mean_win_rate)
        self._append_history("drawdowns", max_drawdown)
        self._append_history("trades_per_episode", mean_trades)

        trading = {
            "mean_win_rate": mean_win_rate,
            "mean_win_rate_status": get_status_color(mean_win_rate, THRESHOLDS.win_rate_good, THRESHOLDS.win_rate_ok, True),
            "max_drawdown": max_drawdown,
            "max_drawdown_status": get_status_color(max_drawdown, THRESHOLDS.drawdown_good, THRESHOLDS.drawdown_ok, False),
            "mean_trades": mean_trades,
            "mean_trades_status": get_range_status(mean_trades, 3, 15),
            "total_trades": total_trades,
        }

        # ─────────────────────────────────────────────────────────────
        # QUALITY METRICS
        # ─────────────────────────────────────────────────────────────
        quality_section = raw.get("quality", {})
        mean_r_multiple = self._safe_float(quality_section.get("mean_r_multiple", raw.get("mean_r_multiple", 0)))
        mean_profit_factor = self._safe_float(quality_section.get("mean_profit_factor", raw.get("mean_profit_factor", 0)))
        mean_entry_quality = self._safe_float(quality_section.get("mean_entry_quality", raw.get("mean_entry_quality", 0.5)))

        self._append_history("r_multiples", mean_r_multiple)

        quality = {
            "mean_r_multiple": mean_r_multiple,
            "mean_r_multiple_status": get_status_color(mean_r_multiple, THRESHOLDS.r_mult_good, THRESHOLDS.r_mult_ok, True),
            "mean_profit_factor": mean_profit_factor,
            "mean_profit_factor_status": get_status_color(mean_profit_factor, THRESHOLDS.pf_good, THRESHOLDS.pf_ok, True),
            "mean_entry_quality": mean_entry_quality,
            "mean_entry_quality_status": get_status_color(mean_entry_quality, THRESHOLDS.eq_good, THRESHOLDS.eq_ok, True),
        }

        # ─────────────────────────────────────────────────────────────
        # EXIT DISTRIBUTION
        # ─────────────────────────────────────────────────────────────
        exit_stats_section = raw.get("exit_stats", {})
        exit_distribution = self._safe_dict(exit_stats_section.get("distribution", raw.get("exit_reason_distribution", {})))
        exit_stats = {"distribution": exit_distribution}

        # ─────────────────────────────────────────────────────────────
        # CURRICULUM
        # ─────────────────────────────────────────────────────────────
        curriculum_progress = self._safe_dict(raw.get("curriculum_progress", {}))
        curriculum_detail = self._safe_dict(raw.get("curriculum_detail", {}))
        processed_curriculum_progress = self._process_curriculum_progress(curriculum_progress)

        # ─────────────────────────────────────────────────────────────
        # RECENT DATA FOR CHARTS
        # ─────────────────────────────────────────────────────────────
        recent_rewards = self._safe_list(raw.get("recent_rewards", []))[-100:]
        recent_pnls = self._safe_list(raw.get("recent_pnls", []))[-100:]
        recent_win_rates = self._safe_list(raw.get("recent_win_rates", []))[-100:]
        recent_drawdowns = self._safe_list(raw.get("recent_drawdowns", []))[-100:]
        recent_r_multiples = self._safe_list(raw.get("recent_r_multiples", []))[-100:]

        for r in recent_rewards[-10:]:
            self._append_history("rewards", self._safe_float(r))
        for p in recent_pnls[-10:]:
            self._append_history("pnls", self._safe_float(p))

        stage_history = self._normalize_stage_history(raw.get("stage_history", []))

        return {
            "status": "active",
            "message": "",
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "raw_timestamp": raw.get("timestamp", ""),

            "progress": progress,
            "learning": learning,
            "trading": trading,
            "quality": quality,
            "exit_stats": exit_stats,

            "curriculum_stage": raw.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": raw.get("curriculum_stage_idx", 0),
            "curriculum_progress": processed_curriculum_progress,
            "curriculum_detail": curriculum_detail,
            "stage_history": stage_history,

            # Optional source-of-truth per-stage requirements (safe fallback if unavailable)
            "requirements_by_stage": get_requirements_by_stage(),

            "recent_rewards": recent_rewards,
            "recent_pnls": recent_pnls,
            "recent_win_rates": recent_win_rates,
            "recent_drawdowns": recent_drawdowns,
            "recent_r_multiples": recent_r_multiples,

            "history": {k: v[-50:] for k, v in self._history.items()},
        }

    def _process_curriculum_progress(self, curriculum_progress: Dict[str, Any]) -> Dict[str, Any]:
        if not curriculum_progress:
            return {}

        result: Dict[str, Any] = {}

        # Basic fields
        result["current_stage"] = curriculum_progress.get("current_stage", "")
        result["stage_index"] = curriculum_progress.get("stage_index", 0)
        result["stage_epoch"] = curriculum_progress.get("stage_epoch", 0)
        result["stage_episodes"] = curriculum_progress.get("stage_episodes", 0)
        result["stage_timesteps"] = curriculum_progress.get("stage_timesteps", 0)
        result["total_episodes"] = curriculum_progress.get("total_episodes", 0)
        result["total_timesteps"] = curriculum_progress.get("total_timesteps", 0)
        result["is_in_transition"] = curriculum_progress.get("is_in_transition", False)
        result["reward_blend_factor"] = curriculum_progress.get("reward_blend_factor", 1.0)
        result["lr_multiplier"] = curriculum_progress.get("lr_multiplier", 1.0)

        # Rolling stats
        result["rolling_stats"] = self._safe_dict(curriculum_progress.get("rolling_stats", {}))

        # Promotion checks (normalized)
        raw_checks = self._safe_dict(curriculum_progress.get("promotion_checks", {}))
        promotion_checks = self._normalize_promotion_checks(raw_checks)
        result["promotion_checks"] = promotion_checks

        # Compute prerequisites (prevents “insufficient data” confusion)
        prereq_keys = ["min_episodes", "min_timesteps", "data_sufficiency"]
        prereqs = []
        for k in prereq_keys:
            c = promotion_checks.get(k)
            if c and c.get("required") is not None:
                prereqs.append({"key": k, "passed": bool(c.get("passed", False)), "required": c.get("required"), "actual": c.get("actual")})
        prereq_passed = sum(1 for p in prereqs if p["passed"])
        result["prerequisites"] = {
            "total": len(prereqs),
            "passed": prereq_passed,
            "all_passed": (len(prereqs) > 0 and prereq_passed == len(prereqs)),
            "items": prereqs,
        }

        # v2.0 Components
        skill_assessment = curriculum_progress.get("skill_assessment")
        if isinstance(skill_assessment, dict):
            result["skill_assessment"] = {
                "scores": self._safe_dict(skill_assessment.get("scores", {})),
                "confidence": self._safe_dict(skill_assessment.get("confidence", {})),
                "weakest_skills": self._safe_list(skill_assessment.get("weakest_skills", [])),
                "strongest_skills": self._safe_list(skill_assessment.get("strongest_skills", [])),
                "requirements_met": self._safe_bool(skill_assessment.get("requirements_met", False)),
            }

        composite_score = curriculum_progress.get("composite_score")
        if isinstance(composite_score, dict):
            base_ready = self._safe_bool(composite_score.get("promotion_ready", False))
            hard_floors = self._safe_bool(composite_score.get("meets_hard_floors", False))
            strict_ready = bool(base_ready and result["prerequisites"]["all_passed"])

            result["composite_score"] = {
                "total_score": self._safe_float(composite_score.get("total_score", 0)),
                "meets_hard_floors": hard_floors,
                "promotion_ready": base_ready,                 # what curriculum says
                "promotion_ready_strict": strict_ready,        # what UI should treat as final truth
                "components": self._safe_dict(composite_score.get("components", {})),
            }

        learning_velocity = curriculum_progress.get("learning_velocity")
        if isinstance(learning_velocity, dict):
            result["learning_velocity"] = {
                "improvement_rate": self._safe_float(learning_velocity.get("improvement_rate", 0)),
                "is_plateaued": self._safe_bool(learning_velocity.get("is_plateaued", False)),
                "plateau_episodes": self._safe_int(learning_velocity.get("plateau_episodes", 0)),
                "per_metric_slopes": self._safe_dict(learning_velocity.get("per_metric_slopes", {})),
                "window_size": self._safe_int(learning_velocity.get("window_size", 100)),
            }

        recovery_protocol = curriculum_progress.get("recovery_protocol")
        if isinstance(recovery_protocol, dict):
            result["recovery_protocol"] = {
                "is_active": self._safe_bool(recovery_protocol.get("is_active", False)),
                "focus_skill": recovery_protocol.get("focus_skill"),
                "episodes_remaining": self._safe_int(recovery_protocol.get("episodes_remaining", 0)),
                "trigger_reason": recovery_protocol.get("trigger_reason", ""),
            }

        review_session = curriculum_progress.get("review_session")
        if isinstance(review_session, dict):
            result["review_session"] = {
                "is_active": self._safe_bool(review_session.get("is_active", False)),
                "review_stage": self._safe_int(review_session.get("review_stage", 0)),
                "home_stage": self._safe_int(review_session.get("home_stage", 0)),
                "episodes_remaining": self._safe_int(review_session.get("episodes_remaining", 0)),
            }

        demotion_analysis = curriculum_progress.get("demotion_analysis")
        if isinstance(demotion_analysis, dict):
            result["demotion_analysis"] = {
                "total_demotions": self._safe_int(demotion_analysis.get("total_demotions", 0)),
                "repeated_failures": self._safe_int(demotion_analysis.get("repeated_failures", 0)),
                "common_failure_reasons": self._safe_list(demotion_analysis.get("common_failure_reasons", [])),
                "weak_skills": self._safe_list(demotion_analysis.get("weak_skills", [])),
            }

        adaptive_thresholds = curriculum_progress.get("adaptive_thresholds")
        if isinstance(adaptive_thresholds, dict):
            result["adaptive_thresholds"] = {
                "relaxation_amount": self._safe_float(adaptive_thresholds.get("relaxation_amount", 0)),
                "max_relaxation": self._safe_float(adaptive_thresholds.get("max_relaxation", 0)),
                "relaxed_metrics": self._safe_list(adaptive_thresholds.get("relaxed_metrics", [])),
            }

        entropy_status = curriculum_progress.get("entropy_status")
        if isinstance(entropy_status, dict):
            result["entropy_status"] = {
                "current": self._safe_float(entropy_status.get("current", 0)),
                "min_target": self._safe_float(entropy_status.get("min_target", 0)),
                "max_target": self._safe_float(entropy_status.get("max_target", 1)),
                "penalty": self._safe_float(entropy_status.get("penalty", 0)),
            }

        result["blockers"] = self._safe_list(curriculum_progress.get("blockers", []))
        result["recommendations"] = self._safe_list(curriculum_progress.get("recommendations", []))
        result["estimated_episodes_to_promotion"] = curriculum_progress.get("estimated_episodes_to_promotion")

        return result

    def _estimate_eta(self, current: int, total: int, raw: Dict) -> float:
        fps = self._safe_float(raw.get("fps", raw.get("learning", {}).get("fps", 0)))
        if fps <= 0 or current <= 0:
            return -1
        remaining = total - current
        return remaining / fps


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if WEB_AVAILABLE:
    app = FastAPI(
        title="PropFirm PPO Training Dashboard v2.1",
        description="Real-time monitoring for PPO trading agent with Curriculum Learning v2.x",
        version="2.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _metrics_reader: Optional[MetricsReader] = None
    _connected_clients: Set[WebSocket] = set()
    _config: DashboardConfig = DashboardConfig()

    def get_metrics_reader() -> MetricsReader:
        global _metrics_reader
        if _metrics_reader is None:
            _metrics_reader = MetricsReader(_config.metrics_file, _config.max_history_points)
        return _metrics_reader

    @app.get("/")
    async def serve_frontend():
        frontend_path = Path(__file__).parent / "index.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return HTMLResponse("<h1>Dashboard frontend not found</h1>", status_code=404)

    @app.get("/api/metrics")
    async def get_metrics():
        reader = get_metrics_reader()
        return JSONResponse(reader.read_metrics())

    @app.get("/api/health")
    async def health_check():
        reader = get_metrics_reader()
        metrics = reader.read_metrics()
        return {
            "status": "healthy",
            "version": "2.1.0",
            "training_active": metrics.get("status") == "active",
            "metrics_file": str(_config.metrics_file),
            "connected_clients": len(_connected_clients),
        }

    @app.get("/api/config")
    async def get_config():
        return asdict(_config)

    @app.get("/api/curriculum")
    async def get_curriculum():
        reader = get_metrics_reader()
        metrics = reader.read_metrics()
        return JSONResponse({
            "curriculum_stage": metrics.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": metrics.get("curriculum_stage_idx", 0),
            "curriculum_progress": metrics.get("curriculum_progress", {}),
            "curriculum_detail": metrics.get("curriculum_detail", {}),
            "stage_history": metrics.get("stage_history", []),
        })

    @app.get("/api/requirements")
    async def get_requirements():
        return JSONResponse(get_requirements_by_stage())

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        _connected_clients.add(websocket)
        logger.info(f"Client connected. Total: {len(_connected_clients)}")

        try:
            reader = get_metrics_reader()
            await websocket.send_json(reader.read_metrics())

            while True:
                await websocket.send_json(reader.read_metrics())
                await asyncio.sleep(_config.update_interval)

        except WebSocketDisconnect:
            logger.info("Client disconnected normally")
        except Exception as e:
            logger.warning(f"WebSocket error: {e}")
        finally:
            _connected_clients.discard(websocket)
            logger.info(f"Client removed. Total: {len(_connected_clients)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

_server_thread: Optional[threading.Thread] = None
_server_running: bool = False


def _run_server(host: str, port: int):
    if not WEB_AVAILABLE:
        logger.error("FastAPI/uvicorn not available")
        return
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)
    server.run()


def start_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    metrics_file: str = "logs/training/live_metrics.json",
    background: bool = True,
) -> Optional[threading.Thread]:
    global _server_thread, _server_running, _config, _metrics_reader

    if not WEB_AVAILABLE:
        logger.error("Cannot start server: FastAPI/uvicorn not installed")
        return None

    if _server_running and _server_thread and _server_thread.is_alive():
        logger.info(f"Dashboard already running at http://localhost:{_config.port}")
        return _server_thread

    _config = DashboardConfig(host=host, port=port, metrics_file=metrics_file)
    _metrics_reader = MetricsReader(metrics_file)

    print()
    print("=" * 70)
    print("  🚀 PROPFIRM PPO TRAINING DASHBOARD v2.1")
    print("=" * 70)
    print(f"  📊 Open http://localhost:{port} in your browser")
    print(f"  📁 Reading metrics from: {metrics_file}")
    print("  ✨ Fixes: strict promotion readiness • normalized gates • requirements export")
    print("=" * 70)
    print()

    if background:
        _server_thread = threading.Thread(target=_run_server, args=(host, port), daemon=True, name="DashboardServer")
        _server_thread.start()
        _server_running = True
        time.sleep(0.5)
        return _server_thread

    _run_server(host, port)
    return None


def stop_dashboard_server():
    global _server_running
    _server_running = False
    logger.info("Dashboard server stopping...")


def main():
    parser = argparse.ArgumentParser(
        description="PropFirm PPO Training Dashboard Server v2.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py
  python server.py --port 8080
  python server.py --metrics-file custom.json
        """,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--metrics-file", default="logs/training/live_metrics.json", help="Path to live_metrics.json")

    args = parser.parse_args()
    start_dashboard_server(host=args.host, port=args.port, metrics_file=args.metrics_file, background=False)


if __name__ == "__main__":
    main()
