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

import numpy as np

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
    """
    Determine status for a value that should be within a target range.
    - good: within [good_min, good_max]
    - ok: slightly outside (within 50% of range width on either side)
    - bad: far outside the target range
    """
    if good_min <= value <= good_max:
        return "good"
    
    range_width = good_max - good_min
    ok_buffer = range_width * 0.5  # 50% buffer on each side for "ok"
    
    # Check if slightly below or above
    if value < good_min:
        distance_below = good_min - value
        return "ok" if distance_below <= ok_buffer else "bad"
    else:  # value > good_max
        distance_above = value - good_max
        return "ok" if distance_above <= ok_buffer else "bad"


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
        from envs.curriculum import CurriculumStage, get_stage_config  # type: ignore

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

    def _normalize_drawdown_percent(self, raw: Dict[str, Any], trading: Dict[str, Any]) -> float:
        """Return max drawdown as percent (0-100).

        The metrics writer can emit multiple drawdown fields:
        - root-level `max_drawdown`: usually a fraction (0..1)
        - `recent_drawdowns`: usually fractions (0..1)
        - `trading.max_drawdown`: sometimes mis-scaled (e.g. 0.681 meaning $681 loss, not 68%)

        For prop-firm style constraints, fractional DD above 50% is implausible.
        Prefer plausible fractional sources when available.
        """
        root_dd = self._safe_float(raw.get("max_drawdown", 0.0))
        trading_dd = self._safe_float(trading.get("max_drawdown", 0.0))

        recent_dds = self._safe_list(raw.get("recent_drawdowns", []))
        recent_last = self._safe_float(recent_dds[-1], 0.0) if recent_dds else 0.0

        # Prefer plausible fraction values first.
        for candidate in (root_dd, recent_last, trading_dd):
            if 0.0 < candidate <= 0.5:
                return candidate * 100.0

        # Next, accept already-percent values (e.g. 6.19 meaning 6.19%).
        for candidate in (root_dd, trading_dd):
            if 1.0 < candidate <= 100.0:
                return candidate

        # Last resort: if a fraction exists (even if large), scale it.
        for candidate in (root_dd, recent_last, trading_dd):
            if 0.0 < candidate <= 1.0:
                return candidate * 100.0

        return 0.0

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

                # Windows file locking fix: retry with backoff on PermissionError
                raw = None
                for attempt in range(3):
                    try:
                        with open(self.metrics_file, "r", encoding="utf-8") as f:
                            raw = json.load(f)
                        break  # Success
                    except PermissionError:
                        if attempt < 2:
                            import time
                            time.sleep(0.05 * (attempt + 1))  # 50ms, 100ms backoff
                        else:
                            # Return cached data on persistent lock
                            return self._last_data if self._last_data else self._empty("File locked...")
                    except json.JSONDecodeError:
                        # Partial write - return cached
                        return self._last_data if self._last_data else self._empty("Reading metrics...")
                
                if raw is None:
                    return self._last_data if self._last_data else self._empty("Reading metrics...")

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

        max_drawdown = self._normalize_drawdown_percent(raw, t)

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
        
        # v5.5: Consecutive loss tracking for governor panel
        max_consecutive_losses = self._safe_int(q.get("max_consecutive_losses", 0))
        avg_consecutive_losses = self._safe_float(q.get("avg_consecutive_losses", 0))
        consecutive_loss_streak_rate = self._safe_float(q.get("consecutive_loss_streak_rate", 0))

        self._append_history("r_multiples", mean_r_multiple)

        quality = {
            "mean_r_multiple": mean_r_multiple,
            "mean_r_multiple_status": get_status_color(mean_r_multiple, THRESHOLDS.r_mult_good, THRESHOLDS.r_mult_ok, True),
            "mean_profit_factor": mean_profit_factor,
            "mean_profit_factor_status": get_status_color(mean_profit_factor, THRESHOLDS.pf_good, THRESHOLDS.pf_ok, True),
            "mean_entry_quality": mean_entry_quality,
            "mean_entry_quality_status": get_status_color(mean_entry_quality, THRESHOLDS.eq_good, THRESHOLDS.eq_ok, True),
            # v5.5: Consecutive loss metrics for governor panel
            "max_consecutive_losses": max_consecutive_losses,
            "avg_consecutive_losses": avg_consecutive_losses,
            "consecutive_loss_streak_rate": consecutive_loss_streak_rate,
        }

        # Exit distribution
        exit_stats = self._safe_dict(raw.get("exit_stats", {}))
        exit_distribution = self._safe_dict(exit_stats.get("distribution", raw.get("exit_reason_distribution", {})))
        exit_stats_out = {"distribution": exit_distribution}

        # Reward component breakdown (new - for market structure signals dashboard)
        reward_components_raw = self._safe_dict(raw.get("reward_components", {}))
        reward_components = self._process_reward_components(reward_components_raw)

        # Stage comparison data for Stage Progress tab
        stage_comparison_raw = self._safe_dict(raw.get("stage_comparison", {}))
        stage_comparison = self._process_stage_comparison(stage_comparison_raw)

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

        # Direction stats (buy/sell breakdown)
        direction_stats = self._safe_dict(raw.get("direction_stats", {}))

        # v5.5: Governor state (loss layer & session budget)
        governor = self._safe_dict(raw.get("governor", {}))

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
            "reward_components": reward_components,
            "stage_comparison": stage_comparison,
            "direction_stats": direction_stats,
            "governor": governor,  # v5.5: Loss layer & session budget state

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

        # Phase 2.2: Regime skill assessment
        if isinstance(cp.get("regime_assessment"), dict):
            ra = cp["regime_assessment"]
            out["regime_assessment"] = {
                "status": ra.get("status", "unknown"),
                "total_trades": self._safe_int(ra.get("total_trades", 0)),
                "min_required": self._safe_int(ra.get("min_required", 50)),
                "confidence": self._safe_float(ra.get("confidence", 0)),
                "regime_coverage": self._safe_float(ra.get("regime_coverage", 0)),
                "scores": self._safe_dict(ra.get("scores", {})),
                "weaknesses": self._safe_list(ra.get("weaknesses", [])),
                "volatility_breakdown": self._safe_dict(ra.get("volatility_breakdown", {})),
                "trend_breakdown": self._safe_dict(ra.get("trend_breakdown", {})),
                "session_breakdown": self._safe_dict(ra.get("session_breakdown", {})),
                "spread_breakdown": self._safe_dict(ra.get("spread_breakdown", {})),
            }

        # Validation gate history and status
        if isinstance(cp.get("validation_gate_history"), list):
            out["validation_gate_history"] = self._safe_list(cp.get("validation_gate_history", []))
        
        if isinstance(cp.get("last_validation_gate"), dict):
            vg = cp["last_validation_gate"]
            out["last_validation_gate"] = {
                "stage": vg.get("stage", ""),
                "stage_epoch": self._safe_int(vg.get("stage_epoch", 0)),
                "stage_episodes": self._safe_int(vg.get("stage_episodes", 0)),
                "timestamp": vg.get("timestamp", ""),
                "passed": self._safe_bool(vg.get("passed", False)),
                "pass_rate": self._safe_float(vg.get("pass_rate", 0)),
                "scenarios_passed": self._safe_int(vg.get("scenarios_passed", 0)),
                "scenarios_total": self._safe_int(vg.get("scenarios_total", 0)),
                "performance_ratio": self._safe_float(vg.get("performance_ratio", 0)),
                "blocking_reasons": self._safe_list(vg.get("blocking_reasons", [])),
            }

        # Stress test history and status
        if isinstance(cp.get("stress_test_history"), list):
            out["stress_test_history"] = self._safe_list(cp.get("stress_test_history", []))
        
        if isinstance(cp.get("last_stress_test"), dict):
            st = cp["last_stress_test"]
            out["last_stress_test"] = {
                "stage": st.get("stage", ""),
                "stage_epoch": self._safe_int(st.get("stage_epoch", 0)),
                "stage_episodes": self._safe_int(st.get("stage_episodes", 0)),
                "timestamp": st.get("timestamp", ""),
                "passed": self._safe_bool(st.get("passed", False)),
                "robustness_score": self._safe_float(st.get("robustness_score", 0)),
                "min_required": self._safe_float(st.get("min_required", 0)),
                "scenarios_count": self._safe_int(st.get("scenarios_count", 0)),
                "summary": self._safe_dict(st.get("summary", {})),
            }

        out["blockers"] = self._safe_list(cp.get("blockers", []))
        out["recommendations"] = self._safe_list(cp.get("recommendations", []))
        out["estimated_episodes_to_promotion"] = cp.get("estimated_episodes_to_promotion")
        out["phase_info"] = cp.get("phase_info", {})  # UI uses it if available

        return out

    def _process_reward_components(self, rc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process reward components into dashboard-friendly format.
        Groups components into categories: market_structure, divergence, regime, exit_quality, etc.
        """
        if not rc:
            return {
                "market_structure": {},
                "divergence": {},
                "regime": {},
                "exit_quality": {},
                "timing": {},
                "risk": {},
                "other": {},
                "summary": {"total_positive": 0, "total_negative": 0, "net": 0},
            }

        # Component category mapping
        market_structure_keys = ["sr_support_bonus", "sr_resistance_bonus", "sr_bad_entry_penalty", 
                                 "structure_alignment_bonus", "bos_alignment_bonus", "order_block_bonus"]
        divergence_keys = ["divergence_contra_penalty", "divergence_aligned_bonus", 
                          "overbought_long_penalty", "oversold_short_penalty"]
        regime_keys = ["risk_off_penalty", "high_vol_penalty"]
        exit_quality_keys = ["exit_quality", "premature_close_penalty", "trailing_stop_bonus"]
        timing_keys = ["off_hours_penalty", "prime_hours_bonus", "time_efficiency", "time_penalty"]
        risk_keys = ["dd_shaping", "churn_penalty", "win_streak_bonus", "loss_streak_penalty"]

        result = {
            "market_structure": {},
            "divergence": {},
            "regime": {},
            "exit_quality": {},
            "timing": {},
            "risk": {},
            "other": {},
        }

        total_positive = 0.0
        total_negative = 0.0

        for name, data in rc.items():
            if isinstance(data, dict):
                total = self._safe_float(data.get("total", 0))
                count = self._safe_int(data.get("count", 0))
                avg = self._safe_float(data.get("avg", 0))
            else:
                total = self._safe_float(data)
                count = 1
                avg = total

            comp_data = {"total": total, "count": count, "avg": avg}

            # Track totals
            if total > 0:
                total_positive += total
            else:
                total_negative += total

            # Categorize
            if name in market_structure_keys:
                result["market_structure"][name] = comp_data
            elif name in divergence_keys:
                result["divergence"][name] = comp_data
            elif name in regime_keys:
                result["regime"][name] = comp_data
            elif name in exit_quality_keys:
                result["exit_quality"][name] = comp_data
            elif name in timing_keys:
                result["timing"][name] = comp_data
            elif name in risk_keys:
                result["risk"][name] = comp_data
            else:
                result["other"][name] = comp_data

        result["summary"] = {
            "total_positive": total_positive,
            "total_negative": total_negative,
            "net": total_positive + total_negative,
        }

        return result

    def _process_stage_comparison(self, sc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process stage comparison data into dashboard-friendly format.
        Provides per-stage metrics and improvements between stages.
        """
        if not sc or not sc.get("stages"):
            return {
                "stages": [],
                "overall_improvement": {},
                "current_stage": "N/A",
                "total_stages_visited": 0,
                "summary": {},
            }

        stages = self._safe_list(sc.get("stages", []))
        overall_improvement = self._safe_dict(sc.get("overall_improvement", {}))
        current_stage = sc.get("current_stage", "N/A")
        total_stages_visited = self._safe_int(sc.get("total_stages_visited", 0))

        # Process each stage
        processed_stages = []
        for stage in stages:
            if not isinstance(stage, dict):
                continue

            processed_stage = {
                "stage_name": stage.get("stage_name", "Unknown"),
                "stage_index": self._safe_int(stage.get("stage_index", 0)),
                "episodes": self._safe_int(stage.get("episodes", 0)),
                "timesteps": self._safe_int(stage.get("timesteps", 0)),
                "total_trades": self._safe_int(stage.get("total_trades", 0)),
                "total_pnl": self._safe_float(stage.get("total_pnl", 0)),
                "avg_pnl": self._safe_float(stage.get("avg_pnl", 0)),
                "pnl_std": self._safe_float(stage.get("pnl_std", 0)),
                "win_rate": self._safe_float(stage.get("win_rate", 0)),
                "win_rate_std": self._safe_float(stage.get("win_rate_std", 0)),
                "avg_drawdown": self._safe_float(stage.get("avg_drawdown", 0)),
                "avg_trades": self._safe_float(stage.get("avg_trades", 0)),
                "avg_reward": self._safe_float(stage.get("avg_reward", 0)),
                "avg_profit_factor": self._safe_float(stage.get("avg_profit_factor", 0)),
                "avg_r_multiple": self._safe_float(stage.get("avg_r_multiple", 0)),
                "first_episode": self._safe_int(stage.get("first_episode", 0)),
                "last_episode": self._safe_int(stage.get("last_episode", 0)),
            }

            # Process direction stats (buy/sell breakdown per stage)
            raw_dir_stats = stage.get("direction_stats")
            if raw_dir_stats and isinstance(raw_dir_stats, dict):
                processed_stage["direction_stats"] = {
                    "long_count": self._safe_int(raw_dir_stats.get("long_count", 0)),
                    "short_count": self._safe_int(raw_dir_stats.get("short_count", 0)),
                    "long_wins": self._safe_int(raw_dir_stats.get("long_wins", 0)),
                    "short_wins": self._safe_int(raw_dir_stats.get("short_wins", 0)),
                    "long_pnl": self._safe_float(raw_dir_stats.get("long_pnl", 0)),
                    "short_pnl": self._safe_float(raw_dir_stats.get("short_pnl", 0)),
                    "long_win_rate": self._safe_float(raw_dir_stats.get("long_win_rate", 0)),
                    "short_win_rate": self._safe_float(raw_dir_stats.get("short_win_rate", 0)),
                    "long_pct": self._safe_float(raw_dir_stats.get("long_pct", 50)),
                    "short_pct": self._safe_float(raw_dir_stats.get("short_pct", 50)),
                    "direction_ratio": self._safe_float(raw_dir_stats.get("direction_ratio", 1.0)),
                }

            # Process improvement data
            improvement = stage.get("improvement")
            if improvement is not None and isinstance(improvement, dict):
                processed_stage["improvement"] = {
                    "win_rate_delta": self._safe_float(improvement.get("win_rate_delta", 0)),
                    "pnl_delta": self._safe_float(improvement.get("pnl_delta", 0)),
                    "profit_factor_delta": self._safe_float(improvement.get("profit_factor_delta", 0)),
                    "reward_delta": self._safe_float(improvement.get("reward_delta", 0)),
                    # Status indicators
                    "win_rate_status": "good" if improvement.get("win_rate_delta", 0) > 0 else "bad",
                    "pnl_status": "good" if improvement.get("pnl_delta", 0) > 0 else "bad",
                    "profit_factor_status": "good" if improvement.get("profit_factor_delta", 0) > 0 else "bad",
                }
            else:
                processed_stage["improvement"] = None

            # Add overall status for the stage
            win_rate = processed_stage["win_rate"]
            avg_pnl = processed_stage["avg_pnl"]
            profit_factor = processed_stage["avg_profit_factor"]
            
            processed_stage["status"] = {
                "win_rate": "good" if win_rate >= 55 else "ok" if win_rate >= 45 else "bad",
                "pnl": "good" if avg_pnl > 0 else "ok" if avg_pnl > -500 else "bad",
                "profit_factor": "good" if profit_factor >= 1.2 else "ok" if profit_factor >= 0.9 else "bad",
            }

            processed_stages.append(processed_stage)

        # Calculate summary statistics across all stages
        summary = {}
        if processed_stages:
            all_win_rates = [s["win_rate"] for s in processed_stages]
            all_pnls = [s["total_pnl"] for s in processed_stages]
            all_pfs = [s["avg_profit_factor"] for s in processed_stages if s["avg_profit_factor"] > 0]
            
            summary = {
                "best_stage_win_rate": max(processed_stages, key=lambda x: x["win_rate"])["stage_name"] if processed_stages else "N/A",
                "best_stage_pnl": max(processed_stages, key=lambda x: x["total_pnl"])["stage_name"] if processed_stages else "N/A",
                "worst_stage_pnl": min(processed_stages, key=lambda x: x["total_pnl"])["stage_name"] if processed_stages else "N/A",
                "avg_win_rate_all_stages": float(np.mean(all_win_rates)) if all_win_rates else 0,
                "total_pnl_all_stages": sum(all_pnls),
                "avg_profit_factor_all_stages": float(np.mean(all_pfs)) if all_pfs else 0,
                "total_episodes_all_stages": sum(s["episodes"] for s in processed_stages),
                "total_trades_all_stages": sum(s["total_trades"] for s in processed_stages),
            }

        return {
            "stages": processed_stages,
            "overall_improvement": {
                "win_rate_delta": self._safe_float(overall_improvement.get("win_rate_delta", 0)),
                "pnl_delta": self._safe_float(overall_improvement.get("pnl_delta", 0)),
                "profit_factor_delta": self._safe_float(overall_improvement.get("profit_factor_delta", 0)),
                "reward_delta": self._safe_float(overall_improvement.get("reward_delta", 0)),
                "stages_progressed": self._safe_int(overall_improvement.get("stages_progressed", 0)),
            },
            "current_stage": current_stage,
            "total_stages_visited": total_stages_visited,
            "summary": summary,
        }

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
