#!/usr/bin/env python3
"""
PropFirm PPO Training Dashboard Server v2.2
- Keeps v2.1 behavior
- Preserves unknown/new promotion check fields (future-proof)
- Tightens structure, reduces duplication
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING

# Ensure project root is importable (dashboard/ is typically one level below root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("dashboard")


# ───────────────────────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────────────────────
@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 8765
    metrics_file: str = "logs/training/live_metrics.json"
    alerts_log_file: str = "logs/audit/alerts.jsonl"
    update_interval: float = 0.5
    max_history_points: int = 500
    max_alert_history: int = 1000


@dataclass
class MetricThresholds:
    win_rate_good: float = 55.0
    win_rate_ok: float = 45.0
    drawdown_good: float = 3.0
    drawdown_ok: float = 6.0
    ev_good: float = 0.5
    ev_ok: float = 0.2
    entropy_good_min: float = 0.25  # PPO entropy is positive
    entropy_good_max: float = 0.85  # Typical healthy range 0.3-0.8
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
    mid = (good_min + good_max) / 2
    # "ok" if relatively close to target band, else bad
    return "ok" if abs(value - mid) <= (good_max - good_min) else "bad"


# ───────────────────────────────────────────────────────────────────────────────
# REQUIREMENTS EXPORT (optional)
# ───────────────────────────────────────────────────────────────────────────────
def _safe_asdict(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [_safe_asdict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _safe_asdict(v) for k, v in obj.items()}
    try:
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
    competence = cfg_dict.get("competence", {}) if isinstance(cfg_dict, dict) else {}

    skill_requirements = {}
    entropy_targets = {}
    composite = {}
    promotion_criteria: Dict[str, Any] = {}

    if isinstance(cfg_dict, dict):
        skill_requirements = cfg_dict.get("skill_requirements") or cfg_dict.get("skills") or cfg_dict.get("skill_thresholds") or {}
        entropy_targets = cfg_dict.get("entropy_targets") or cfg_dict.get("entropy") or {}
        composite = cfg_dict.get("composite_scoring") or cfg_dict.get("composite") or cfg_dict.get("composite_score") or {}

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

            stages_out[stage_name] = {"stage_index": stage_idx, "requirements": _extract_stage_requirements(cfg)}

        payload["available"] = True
        payload["source"] = "envs.curriculum_config.get_stage_config"
        payload["stages"] = stages_out
        return payload

    except Exception as e:
        payload["error"] = f"{type(e).__name__}: {e}"
        return payload


# ───────────────────────────────────────────────────────────────────────────────
# METRICS READER
# ───────────────────────────────────────────────────────────────────────────────
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

    @staticmethod
    def _is_nan(x: float) -> bool:
        return x != x

    def _safe_float(self, val: Any, default: float = 0.0) -> float:
        if val is None:
            return default
        try:
            f = float(val)
            return default if self._is_nan(f) else f
        except (TypeError, ValueError):
            return default

    def _safe_int(self, val: Any, default: int = 0) -> int:
        return int(self._safe_float(val, float(default)))

    def _safe_list(self, val: Any) -> List[Any]:
        return val if isinstance(val, list) else []

    def _safe_dict(self, val: Any) -> Dict[str, Any]:
        return val if isinstance(val, dict) else {}

    def _safe_bool(self, val: Any, default: bool = False) -> bool:
        return val if isinstance(val, bool) else default

    def _append_history(self, key: str, value: float) -> None:
        if key not in self._history:
            return
        self._history[key].append(value)
        if len(self._history[key]) > self.max_history:
            self._history[key] = self._history[key][-self.max_history:]

    def read_metrics(self) -> Dict[str, Any]:
        with self._lock:
            if not self.metrics_file.exists():
                return self._empty("Waiting for training to start...")

            try:
                mtime = self.metrics_file.stat().st_mtime
                if mtime == self._last_modified and self._last_data:
                    return self._last_data

                with open(self.metrics_file, "r", encoding="utf-8") as f:
                    raw = json.load(f)

                self._last_modified = mtime
                processed = self._process(raw)
                self._last_data = processed
                return processed

            except json.JSONDecodeError:
                return self._last_data if self._last_data else self._empty("Reading metrics...")
            except Exception as e:
                logger.warning(f"Error reading metrics: {type(e).__name__}: {e}")
                return self._last_data if self._last_data else self._empty(str(e))

    def _empty(self, message: str = "") -> Dict[str, Any]:
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
            "recent_r_multiples": [],
            "stage_history": [],
            "requirements_by_stage": get_requirements_by_stage(),
        }

    # ──────────────── normalization helpers ────────────────
    def _normalize_key(self, k: str) -> str:
        k0 = (k or "").strip()
        if not k0:
            return k0
        k1 = k0.lower().replace(" ", "_").replace("-", "_")
        if k1 in ("min_episodes", "episodes", "stage_episodes"):
            return "min_episodes"
        if k1 in ("min_timesteps", "timesteps", "stage_timesteps"):
            return "min_timesteps"
        if k1 in ("insufficient_data", "sufficient_data", "data_sufficiency", "data_sufficient", "insufficientdata"):
            return "data_sufficiency"
        return k1

    def _normalize_promotion_checks(self, checks: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (checks or {}).items():
            if not isinstance(v, dict):
                continue
            nk = self._normalize_key(str(k))

            base = {
                "required": v.get("required"),
                "actual": v.get("actual"),
                "passed": bool(v.get("passed", False)),
                "original_key": k,
            }

            # preserve all extra fields (future-proof)
            for fk, fv in v.items():
                if fk not in base:
                    base[fk] = fv

            out[nk] = base
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

    # ──────────────── core processing ────────────────
    def _process(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        # Progress
        p = self._safe_dict(raw.get("progress", {}))
        timesteps = self._safe_int(p.get("timesteps", raw.get("timesteps", 0)))
        total_timesteps = self._safe_int(p.get("total_timesteps", raw.get("total_timesteps", 1)))
        progress_pct = self._safe_float(p.get("progress_pct", raw.get("progress_pct", 0)))
        total_episodes = self._safe_int(p.get("total_episodes", raw.get("total_episodes", 0)))

        progress = {
            "timesteps": timesteps,
            "total_timesteps": total_timesteps,
            "progress_pct": progress_pct,
            "total_episodes": total_episodes,
            "eta_seconds": self._estimate_eta(timesteps, total_timesteps, raw),
        }

        # Learning
        l = self._safe_dict(raw.get("learning", {}))
        mean_reward = self._safe_float(l.get("mean_reward", raw.get("mean_reward", 0)))
        total_pnl = self._safe_float(l.get("total_pnl", raw.get("total_pnl", 0)))

        approx_kl = self._safe_float(l.get("kl_divergence", l.get("approx_kl", raw.get("approx_kl", 0))))
        clip_fraction = self._safe_float(l.get("clip_fraction", raw.get("clip_fraction", 0)))
        entropy = self._safe_float(l.get("entropy", raw.get("entropy", 0)))
        explained_variance = self._safe_float(l.get("explained_variance", raw.get("explained_variance", 0)))
        value_loss = self._safe_float(l.get("value_loss", raw.get("value_loss", 0)))
        policy_loss = self._safe_float(l.get("policy_loss", raw.get("policy_loss", 0)))
        learning_rate = self._safe_float(l.get("learning_rate", raw.get("learning_rate", 3e-4)))
        fps = self._safe_float(l.get("fps", raw.get("fps", 0)))
        n_updates = self._safe_int(l.get("n_updates", raw.get("n_updates", 0)))
        clip_range = self._safe_float(l.get("clip_range", raw.get("clip_range", 0.2)))

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

        # Trading
        t = self._safe_dict(raw.get("trading", {}))
        mean_win_rate = self._safe_float(t.get("mean_win_rate", raw.get("mean_win_rate", 0)))
        if 0 < mean_win_rate < 1:
            mean_win_rate *= 100

        max_drawdown = self._safe_float(t.get("max_drawdown", raw.get("max_drawdown", 0)))
        if 0 < max_drawdown < 1:
            max_drawdown *= 100

        mean_trades = self._safe_float(t.get("mean_trades", raw.get("mean_trades", 0)))
        total_trades = self._safe_int(t.get("total_trades", raw.get("total_trades", 0)))

        # Note: win_rates, drawdowns, r_multiples history is populated from recent_* arrays below
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

        # Quality
        q = self._safe_dict(raw.get("quality", {}))
        mean_r_multiple = self._safe_float(q.get("mean_r_multiple", raw.get("mean_r_multiple", 0)))
        mean_profit_factor = self._safe_float(q.get("mean_profit_factor", raw.get("mean_profit_factor", 0)))
        mean_entry_quality = self._safe_float(q.get("mean_entry_quality", raw.get("mean_entry_quality", 0.5)))

        self._append_history("r_multiples", mean_r_multiple)

        quality = {
            "mean_r_multiple": mean_r_multiple,
            "mean_r_multiple_status": get_status_color(mean_r_multiple, THRESHOLDS.r_mult_good, THRESHOLDS.r_mult_ok, True),
            "mean_profit_factor": mean_profit_factor,
            "mean_profit_factor_status": get_status_color(mean_profit_factor, THRESHOLDS.pf_good, THRESHOLDS.pf_ok, True),
            "mean_entry_quality": mean_entry_quality,
            "mean_entry_quality_status": get_status_color(mean_entry_quality, THRESHOLDS.eq_good, THRESHOLDS.eq_ok, True),
        }

        # Exit distribution
        exit_stats = self._safe_dict(raw.get("exit_stats", {}))
        exit_distribution = self._safe_dict(exit_stats.get("distribution", raw.get("exit_reason_distribution", {})))
        exit_stats_out = {"distribution": exit_distribution}

        # Curriculum (pass-through + normalized checks)
        curriculum_progress = self._safe_dict(raw.get("curriculum_progress", {}))
        curriculum_detail = self._safe_dict(raw.get("curriculum_detail", {}))
        processed_cp = self._process_curriculum_progress(curriculum_progress)

        # Recent series
        recent_rewards = self._safe_list(raw.get("recent_rewards", []))[-100:]
        recent_pnls = self._safe_list(raw.get("recent_pnls", []))[-100:]
        recent_win_rates = self._safe_list(raw.get("recent_win_rates", []))[-100:]
        recent_drawdowns = self._safe_list(raw.get("recent_drawdowns", []))[-100:]
        recent_r_multiples = self._safe_list(raw.get("recent_r_multiples", []))[-100:]

        # Append recent values to history arrays for slope/volatility calculations
        for r in recent_rewards[-10:]:
            self._append_history("rewards", self._safe_float(r))
        for p2 in recent_pnls[-10:]:
            self._append_history("pnls", self._safe_float(p2))
        for wr in recent_win_rates[-10:]:
            self._append_history("win_rates", self._safe_float(wr))
        for dd in recent_drawdowns[-10:]:
            self._append_history("drawdowns", self._safe_float(dd))
        for rm in recent_r_multiples[-10:]:
            self._append_history("r_multiples", self._safe_float(rm))

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
            "exit_stats": exit_stats_out,

            "curriculum_stage": raw.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": raw.get("curriculum_stage_idx", 0),
            "curriculum_progress": processed_cp,
            "curriculum_detail": curriculum_detail,
            "stage_history": stage_history,

            "requirements_by_stage": get_requirements_by_stage(),

            "recent_rewards": recent_rewards,
            "recent_pnls": recent_pnls,
            "recent_win_rates": recent_win_rates,
            "recent_drawdowns": recent_drawdowns,
            "recent_r_multiples": recent_r_multiples,

            "history": {k: v[-50:] for k, v in self._history.items()},
        }

    def _process_curriculum_progress(self, cp: Dict[str, Any]) -> Dict[str, Any]:
        if not cp:
            return {}

        out: Dict[str, Any] = {
            "current_stage": cp.get("current_stage", ""),
            "stage_index": cp.get("stage_index", 0),
            "stage_epoch": cp.get("stage_epoch", 0),
            "stage_episodes": cp.get("stage_episodes", 0),
            "stage_timesteps": cp.get("stage_timesteps", 0),
            "total_episodes": cp.get("total_episodes", 0),
            "total_timesteps": cp.get("total_timesteps", 0),
            "is_in_transition": cp.get("is_in_transition", False),
            "reward_blend_factor": cp.get("reward_blend_factor", 1.0),
            "lr_multiplier": cp.get("lr_multiplier", 1.0),
            "rolling_stats": self._safe_dict(cp.get("rolling_stats", {})),
        }

        # Checks
        raw_checks = self._safe_dict(cp.get("promotion_checks", {}))
        out["promotion_checks"] = self._normalize_promotion_checks(raw_checks)

        # Prerequisites block (UI friendliness)
        prereq_keys = ["min_episodes", "min_timesteps", "data_sufficiency"]
        prereqs = []
        for k in prereq_keys:
            c = out["promotion_checks"].get(k)
            if c and c.get("required") is not None:
                prereqs.append({"key": k, "passed": bool(c.get("passed", False)), "required": c.get("required"), "actual": c.get("actual")})
        passed = sum(1 for p in prereqs if p["passed"])
        out["prerequisites"] = {"total": len(prereqs), "passed": passed, "all_passed": (len(prereqs) > 0 and passed == len(prereqs)), "items": prereqs}

        # Pass-through components used by UI
        if isinstance(cp.get("skill_assessment"), dict):
            sa = cp["skill_assessment"]
            out["skill_assessment"] = {
                "scores": self._safe_dict(sa.get("skill_scores", sa.get("scores", {}))),
                "confidence": self._safe_dict(sa.get("skill_confidence", sa.get("confidence", {}))),
                "weakest_skills": self._safe_list(sa.get("weakest_skills", [])),
                "strongest_skills": self._safe_list(sa.get("strongest_skills", [])),
                "weighted_average": self._safe_float(sa.get("weighted_average", 0)),
                "requirements_met": self._safe_bool(sa.get("requirements_met", False)),
            }

        if isinstance(cp.get("composite_score"), dict):
            cs = cp["composite_score"]
            base_ready = self._safe_bool(cs.get("promotion_ready", False))
            hard_floors = self._safe_bool(cs.get("meets_hard_floors", False))
            strict_ready = bool(base_ready and out["prerequisites"]["all_passed"])
            out["composite_score"] = {
                "total_score": self._safe_float(cs.get("total_score", 0)),
                "meets_hard_floors": hard_floors,
                "promotion_ready": base_ready,
                "promotion_ready_strict": strict_ready,
                "components": self._safe_dict(cs.get("components", {})),
            }

        if isinstance(cp.get("learning_velocity"), dict):
            lv = cp["learning_velocity"]
            out["learning_velocity"] = {
                "improvement_rate": self._safe_float(lv.get("average_improvement", lv.get("improvement_rate", 0))),
                "is_plateaued": self._safe_bool(lv.get("is_plateaued", False)),
                "plateau_episodes": self._safe_int(lv.get("plateau_episodes", 0)),
                "per_metric_slopes": self._safe_dict(lv.get("improvement_rates", lv.get("per_metric_slopes", {}))),
                "window_size": self._safe_int(lv.get("window_size", 100)),
            }

        if isinstance(cp.get("recovery_protocol"), dict):
            rp = cp["recovery_protocol"]
            out["recovery_protocol"] = {
                "is_active": self._safe_bool(rp.get("is_active", False)),
                "focus_skill": rp.get("focus_skill"),
                "episodes_remaining": self._safe_int(rp.get("episodes_remaining", 0)),
                "trigger_reason": rp.get("trigger_reason", ""),
            }

        if isinstance(cp.get("review_session"), dict):
            rs = cp["review_session"]
            out["review_session"] = {
                "is_active": self._safe_bool(rs.get("is_active", False)),
                "review_stage": self._safe_int(rs.get("review_stage", 0)),
                "home_stage": self._safe_int(rs.get("home_stage", 0)),
                "episodes_remaining": self._safe_int(rs.get("episodes_remaining", 0)),
            }

        if isinstance(cp.get("demotion_analysis"), dict):
            da = cp["demotion_analysis"]
            out["demotion_analysis"] = {
                "total_demotions": self._safe_int(da.get("total_demotions", 0)),
                "repeated_failures": self._safe_int(da.get("repeated_failures", 0)),
                "common_failure_reasons": self._safe_list(da.get("common_failure_reasons", [])),
                "weak_skills": self._safe_list(da.get("weak_skills", [])),
            }

        if isinstance(cp.get("adaptive_thresholds"), dict):
            at = cp["adaptive_thresholds"]
            out["adaptive_thresholds"] = {
                "relaxation_amount": self._safe_float(at.get("relaxation_amount", 0)),
                "max_relaxation": self._safe_float(at.get("max_relaxation", 0)),
                "relaxed_metrics": self._safe_list(at.get("relaxed_metrics", [])),
            }

        if isinstance(cp.get("entropy_status"), dict):
            es = cp["entropy_status"]
            out["entropy_status"] = {
                "current": self._safe_float(es.get("current", 0)),
                "min_target": self._safe_float(es.get("min_target", 0)),
                "max_target": self._safe_float(es.get("max_target", 1)),
                "penalty": self._safe_float(es.get("penalty", 0)),
            }

        out["blockers"] = self._safe_list(cp.get("blockers", []))
        out["recommendations"] = self._safe_list(cp.get("recommendations", []))
        out["estimated_episodes_to_promotion"] = cp.get("estimated_episodes_to_promotion")
        out["phase_info"] = cp.get("phase_info", {})  # UI uses it if available

        return out

    def _estimate_eta(self, current: int, total: int, raw: Dict[str, Any]) -> float:
        fps = self._safe_float(raw.get("fps", raw.get("learning", {}).get("fps", 0)))
        if fps <= 0 or current <= 0:
            return -1
        remaining = total - current
        return remaining / fps


# ───────────────────────────────────────────────────────────────────────────────
# FASTAPI
# ───────────────────────────────────────────────────────────────────────────────
if WEB_AVAILABLE:
    app = FastAPI(
        title="PropFirm PPO Training Dashboard v2.2",
        description="Real-time monitoring for PPO trading agent with Curriculum Learning v2.x",
        version="2.2.0",
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
        return JSONResponse(get_metrics_reader().read_metrics())

    @app.get("/api/health")
    async def health_check():
        metrics = get_metrics_reader().read_metrics()
        return {
            "status": "healthy",
            "version": "2.2.0",
            "training_active": metrics.get("status") == "active",
            "metrics_file": str(_config.metrics_file),
            "connected_clients": len(_connected_clients),
        }

    @app.get("/api/config")
    async def get_config():
        return asdict(_config)

    @app.get("/api/curriculum")
    async def get_curriculum():
        metrics = get_metrics_reader().read_metrics()
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

    # ─────────────────────────────────────────────────────────────────────────
    # ALERT LOGGING API
    # ─────────────────────────────────────────────────────────────────────────
    from fastapi import Request

    @app.post("/api/alerts/log")
    async def log_alert(request: Request):
        """Log an alert to the audit file for persistence."""
        try:
            alert_data = await request.json()
            alert_log_path = Path(_config.alerts_log_file)
            alert_log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Add server timestamp
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "logged_at_epoch": time.time(),
                **alert_data
            }
            
            # Append to JSONL file
            with open(alert_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            logger.info(f"Alert logged: [{alert_data.get('severity', 'unknown')}] {alert_data.get('title', 'untitled')}")
            return JSONResponse({"status": "ok", "logged": True})
        except Exception as e:
            logger.error(f"Failed to log alert: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @app.get("/api/alerts/history")
    async def get_alert_history(limit: int = 100, severity: Optional[str] = None):
        """Retrieve recent alerts from the log file."""
        try:
            alert_log_path = Path(_config.alerts_log_file)
            if not alert_log_path.exists():
                return JSONResponse({"alerts": [], "total": 0})
            
            alerts = []
            with open(alert_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        if severity and entry.get("severity") != severity:
                            continue
                        alerts.append(entry)
                    except json.JSONDecodeError:
                        continue
            
            # Return most recent first
            alerts = alerts[-min(limit, _config.max_alert_history):]
            alerts.reverse()
            
            return JSONResponse({"alerts": alerts, "total": len(alerts)})
        except Exception as e:
            logger.error(f"Failed to read alert history: {e}")
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

    @app.get("/api/alerts/stats")
    async def get_alert_stats():
        """Get alert statistics."""
        try:
            alert_log_path = Path(_config.alerts_log_file)
            if not alert_log_path.exists():
                return JSONResponse({"total": 0, "by_severity": {}, "by_metric": {}})
            
            total = 0
            by_severity = {"critical": 0, "warning": 0, "info": 0}
            by_metric = {}
            
            with open(alert_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        total += 1
                        sev = entry.get("severity", "info")
                        by_severity[sev] = by_severity.get(sev, 0) + 1
                        metric = entry.get("metric", "unknown")
                        by_metric[metric] = by_metric.get(metric, 0) + 1
                    except json.JSONDecodeError:
                        continue
            
            return JSONResponse({
                "total": total,
                "by_severity": by_severity,
                "by_metric": dict(sorted(by_metric.items(), key=lambda x: -x[1])[:20])
            })
        except Exception as e:
            return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

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
            logger.warning(f"WebSocket error: {type(e).__name__}: {e}")
        finally:
            _connected_clients.discard(websocket)
            logger.info(f"Client removed. Total: {len(_connected_clients)}")


def start_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    metrics_file: str = "logs/training/live_metrics.json",
    background: bool = True,
) -> Optional[threading.Thread]:
    if not WEB_AVAILABLE:
        logger.error("Cannot start server: FastAPI/uvicorn not installed")
        return None

    global _config, _metrics_reader
    _config = DashboardConfig(host=host, port=port, metrics_file=metrics_file)
    _metrics_reader = MetricsReader(metrics_file)

    print()
    print("=" * 70)
    print("  🚀 PROPFIRM PPO TRAINING DASHBOARD v2.2")
    print("=" * 70)
    print(f"  📊 Open http://localhost:{port} in your browser")
    print(f"  📁 Reading metrics from: {metrics_file}")
    print("=" * 70)
    print()

    def _run():
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
        uvicorn.Server(config).run()

    if background:
        t = threading.Thread(target=_run, daemon=True, name="DashboardServer")
        t.start()
        time.sleep(0.5)
        return t

    _run()
    return None


def main():
    parser = argparse.ArgumentParser(description="PropFirm PPO Training Dashboard Server v2.2")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument("--metrics-file", default="logs/training/live_metrics.json", help="Path to live_metrics.json")
    args = parser.parse_args()

    if not WEB_AVAILABLE:
        raise SystemExit("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn websockets")

    start_dashboard_server(host=args.host, port=args.port, metrics_file=args.metrics_file, background=False)


if __name__ == "__main__":
    main()
