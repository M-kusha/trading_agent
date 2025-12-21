#!/usr/bin/env python3
"""
PropFirm PPO Training Dashboard Server v2.0
============================================

Real-time dashboard server for monitoring PPO training progress with
full Curriculum Learning v2.0 support.

Features:
- WebSocket streaming for real-time updates
- REST API for snapshot data
- File watching for automatic updates
- Full v2.0 curriculum data extraction:
  - Skill assessment (10 trading skills)
  - Composite scoring with hard floors
  - Learning velocity tracking
  - Recovery protocol status
  - Review session status
  - Demotion analysis
  - Adaptive thresholds
  - Entropy management
  - Recommendations and blockers

Usage:
    Standalone: python server.py [--port 8765] [--metrics-file path/to/live_metrics.json]
    Embedded:   from dashboard.server import start_dashboard_server
                start_dashboard_server(port=8765)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from dataclasses import dataclass, asdict
import argparse
import logging

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
    """Dashboard server configuration."""
    host: str = "0.0.0.0"
    port: int = 8765
    metrics_file: str = "logs/training/live_metrics.json"
    update_interval: float = 0.5  # seconds between WebSocket updates
    file_poll_interval: float = 0.1  # seconds between file checks
    max_history_points: int = 500  # Max data points for charts


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLDS FOR STATUS COLORS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricThresholds:
    """Thresholds for color-coding metrics."""
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


def get_status_color(value: float, good_threshold: float, ok_threshold: float, 
                     higher_is_better: bool = True) -> str:
    """Determine status color based on thresholds."""
    if higher_is_better:
        if value >= good_threshold:
            return "good"
        elif value >= ok_threshold:
            return "ok"
        else:
            return "bad"
    else:
        if value <= good_threshold:
            return "good"
        elif value <= ok_threshold:
            return "ok"
        else:
            return "bad"


def get_range_status(value: float, good_min: float, good_max: float) -> str:
    """Determine status for values that should be in a range."""
    if good_min <= value <= good_max:
        return "good"
    elif abs(value - (good_min + good_max) / 2) < abs(good_max - good_min):
        return "ok"
    else:
        return "bad"


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS READER
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsReader:
    """
    Reads and processes training metrics from live_metrics.json.
    Enhanced for Curriculum v2.0 data structures.
    """
    
    def __init__(self, metrics_file: str, max_history: int = 500):
        self.metrics_file = Path(metrics_file)
        self.max_history = max_history
        self._last_modified: float = 0
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
        """Safely convert to float."""
        if val is None:
            return default
        try:
            f = float(val)
            if f != f:  # NaN check
                return default
            return f
        except (TypeError, ValueError):
            return default
    
    def _safe_int(self, val: Any, default: int = 0) -> int:
        """Safely convert to int."""
        try:
            return int(self._safe_float(val, float(default)))
        except Exception:
            return default
    
    def _safe_list(self, val: Any, default: Optional[List] = None) -> List:
        """Safely get list."""
        if default is None:
            default = []
        if isinstance(val, list):
            return val
        return default
    
    def _safe_dict(self, val: Any, default: Optional[Dict] = None) -> Dict:
        """Safely get dict."""
        if default is None:
            default = {}
        if isinstance(val, dict):
            return val
        return default
    
    def _safe_bool(self, val: Any, default: bool = False) -> bool:
        """Safely get bool."""
        if isinstance(val, bool):
            return val
        return default
    
    def _append_history(self, key: str, value: float) -> None:
        """Append value to history, maintaining max size."""
        if key in self._history:
            self._history[key].append(value)
            if len(self._history[key]) > self.max_history:
                self._history[key] = self._history[key][-self.max_history:]
    
    def read_metrics(self) -> Dict[str, Any]:
        """
        Read and process metrics from file.
        Returns comprehensive dashboard data structure.
        """
        with self._lock:
            if not self.metrics_file.exists():
                return self._get_empty_metrics("Waiting for training to start...")
            
            try:
                mtime = self.metrics_file.stat().st_mtime
                if mtime == self._last_modified and self._last_data:
                    return self._last_data
                
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
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
        """Return empty metrics structure."""
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
        }
    
    def _process_metrics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw metrics into dashboard format with v2.0 curriculum support."""
        
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
        if mean_win_rate < 1 and mean_win_rate > 0:
            mean_win_rate = mean_win_rate * 100
        
        max_drawdown_raw = trading_section.get("max_drawdown", raw.get("max_drawdown", 0))
        max_drawdown = self._safe_float(max_drawdown_raw)
        if max_drawdown < 1 and max_drawdown > 0:
            max_drawdown = max_drawdown * 100
        
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
        exit_distribution = self._safe_dict(
            exit_stats_section.get("distribution", raw.get("exit_reason_distribution", {}))
        )
        
        exit_stats = {
            "distribution": exit_distribution,
        }
        
        # ─────────────────────────────────────────────────────────────
        # CURRICULUM v2.0 DATA
        # ─────────────────────────────────────────────────────────────
        curriculum_progress = self._safe_dict(raw.get("curriculum_progress", {}))
        curriculum_detail = self._safe_dict(raw.get("curriculum_detail", {}))
        
        # Extract v2.0 components from curriculum_progress
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
        
        # ─────────────────────────────────────────────────────────────
        # ASSEMBLE FINAL PAYLOAD
        # ─────────────────────────────────────────────────────────────
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
            
            # Curriculum data
            "curriculum_stage": raw.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": raw.get("curriculum_stage_idx", 0),
            "curriculum_progress": processed_curriculum_progress,
            "curriculum_detail": curriculum_detail,
            "stage_history": raw.get("stage_history", []),
            
            # Chart data
            "recent_rewards": recent_rewards,
            "recent_pnls": recent_pnls,
            "recent_win_rates": recent_win_rates,
            "recent_drawdowns": recent_drawdowns,
            "recent_r_multiples": recent_r_multiples,
            
            # History for sparklines
            "history": {k: v[-50:] for k, v in self._history.items()},
        }
    
    def _process_curriculum_progress(self, curriculum_progress: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process curriculum_progress from get_progress_report() into dashboard format.
        
        Extracts v2.0 components:
        - skill_assessment
        - composite_score
        - learning_velocity
        - recovery_protocol
        - review_session
        - demotion_analysis
        - adaptive_thresholds
        - entropy_status
        - recommendations
        - blockers
        """
        if not curriculum_progress:
            return {}
        
        result = {}
        
        # Pass through basic fields
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
        
        # Promotion checks (criteria)
        result["promotion_checks"] = self._safe_dict(curriculum_progress.get("promotion_checks", {}))
        
        # ─── v2.0 Components ───
        
        # Skill Assessment
        skill_assessment = curriculum_progress.get("skill_assessment")
        if skill_assessment:
            result["skill_assessment"] = {
                "scores": self._safe_dict(skill_assessment.get("scores", {})),
                "confidence": self._safe_dict(skill_assessment.get("confidence", {})),
                "weakest_skills": self._safe_list(skill_assessment.get("weakest_skills", [])),
                "strongest_skills": self._safe_list(skill_assessment.get("strongest_skills", [])),
                "requirements_met": self._safe_bool(skill_assessment.get("requirements_met", False)),
            }
        
        # Composite Score
        composite_score = curriculum_progress.get("composite_score")
        if composite_score:
            result["composite_score"] = {
                "total_score": self._safe_float(composite_score.get("total_score", 0)),
                "meets_hard_floors": self._safe_bool(composite_score.get("meets_hard_floors", False)),
                "promotion_ready": self._safe_bool(composite_score.get("promotion_ready", False)),
                "components": self._safe_dict(composite_score.get("components", {})),
            }
        
        # Learning Velocity
        learning_velocity = curriculum_progress.get("learning_velocity")
        if learning_velocity:
            result["learning_velocity"] = {
                "improvement_rate": self._safe_float(learning_velocity.get("improvement_rate", 0)),
                "is_plateaued": self._safe_bool(learning_velocity.get("is_plateaued", False)),
                "plateau_episodes": self._safe_int(learning_velocity.get("plateau_episodes", 0)),
                "per_metric_slopes": self._safe_dict(learning_velocity.get("per_metric_slopes", {})),
                "window_size": self._safe_int(learning_velocity.get("window_size", 100)),
            }
        
        # Recovery Protocol
        recovery_protocol = curriculum_progress.get("recovery_protocol")
        if recovery_protocol:
            result["recovery_protocol"] = {
                "is_active": self._safe_bool(recovery_protocol.get("is_active", False)),
                "focus_skill": recovery_protocol.get("focus_skill"),
                "episodes_remaining": self._safe_int(recovery_protocol.get("episodes_remaining", 0)),
                "trigger_reason": recovery_protocol.get("trigger_reason", ""),
            }
        
        # Review Session
        review_session = curriculum_progress.get("review_session")
        if review_session:
            result["review_session"] = {
                "is_active": self._safe_bool(review_session.get("is_active", False)),
                "review_stage": self._safe_int(review_session.get("review_stage", 0)),
                "home_stage": self._safe_int(review_session.get("home_stage", 0)),
                "episodes_remaining": self._safe_int(review_session.get("episodes_remaining", 0)),
            }
        
        # Demotion Analysis
        demotion_analysis = curriculum_progress.get("demotion_analysis")
        if demotion_analysis:
            result["demotion_analysis"] = {
                "total_demotions": self._safe_int(demotion_analysis.get("total_demotions", 0)),
                "repeated_failures": self._safe_int(demotion_analysis.get("repeated_failures", 0)),
                "common_failure_reasons": self._safe_list(demotion_analysis.get("common_failure_reasons", [])),
                "weak_skills": self._safe_list(demotion_analysis.get("weak_skills", [])),
            }
        
        # Adaptive Thresholds
        adaptive_thresholds = curriculum_progress.get("adaptive_thresholds")
        if adaptive_thresholds:
            result["adaptive_thresholds"] = {
                "relaxation_amount": self._safe_float(adaptive_thresholds.get("relaxation_amount", 0)),
                "max_relaxation": self._safe_float(adaptive_thresholds.get("max_relaxation", 0)),
                "relaxed_metrics": self._safe_list(adaptive_thresholds.get("relaxed_metrics", [])),
            }
        
        # Entropy Status
        entropy_status = curriculum_progress.get("entropy_status")
        if entropy_status:
            result["entropy_status"] = {
                "current": self._safe_float(entropy_status.get("current", 0)),
                "min_target": self._safe_float(entropy_status.get("min_target", 0)),
                "max_target": self._safe_float(entropy_status.get("max_target", 1)),
                "penalty": self._safe_float(entropy_status.get("penalty", 0)),
            }
        
        # Blockers and Recommendations
        result["blockers"] = self._safe_list(curriculum_progress.get("blockers", []))
        result["recommendations"] = self._safe_list(curriculum_progress.get("recommendations", []))
        
        # Estimated episodes to promotion
        result["estimated_episodes_to_promotion"] = curriculum_progress.get("estimated_episodes_to_promotion")
        
        return result
    
    def _estimate_eta(self, current: int, total: int, raw: Dict) -> float:
        """Estimate time remaining in seconds."""
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
        title="PropFirm PPO Training Dashboard v2.0",
        description="Real-time monitoring for PPO trading agent with Curriculum Learning v2.0",
        version="2.0.0"
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
        """Get or create metrics reader."""
        global _metrics_reader
        if _metrics_reader is None:
            _metrics_reader = MetricsReader(_config.metrics_file, _config.max_history_points)
        return _metrics_reader
    
    
    @app.get("/")
    async def serve_frontend():
        """Serve the frontend HTML."""
        frontend_path = Path(__file__).parent / "index.html"
        if frontend_path.exists():
            return FileResponse(frontend_path)
        return HTMLResponse("<h1>Dashboard frontend not found</h1>", status_code=404)
    
    
    @app.get("/api/metrics")
    async def get_metrics():
        """REST endpoint for current metrics."""
        reader = get_metrics_reader()
        return JSONResponse(reader.read_metrics())
    
    
    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        reader = get_metrics_reader()
        metrics = reader.read_metrics()
        return {
            "status": "healthy",
            "version": "2.0.0",
            "training_active": metrics.get("status") == "active",
            "metrics_file": str(_config.metrics_file),
            "connected_clients": len(_connected_clients),
        }
    
    
    @app.get("/api/config")
    async def get_config():
        """Get dashboard configuration."""
        return asdict(_config)
    
    
    @app.get("/api/curriculum")
    async def get_curriculum():
        """Get curriculum-specific data."""
        reader = get_metrics_reader()
        metrics = reader.read_metrics()
        return JSONResponse({
            "curriculum_stage": metrics.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": metrics.get("curriculum_stage_idx", 0),
            "curriculum_progress": metrics.get("curriculum_progress", {}),
            "curriculum_detail": metrics.get("curriculum_detail", {}),
            "stage_history": metrics.get("stage_history", []),
        })
    
    
    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket for real-time streaming."""
        await websocket.accept()
        _connected_clients.add(websocket)
        logger.info(f"Client connected. Total: {len(_connected_clients)}")
        
        try:
            reader = get_metrics_reader()
            
            # Send initial data
            await websocket.send_json(reader.read_metrics())
            
            # Stream updates
            while True:
                data = reader.read_metrics()
                await websocket.send_json(data)
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
    """Run uvicorn server (blocking)."""
    if not WEB_AVAILABLE:
        logger.error("FastAPI/uvicorn not available")
        return
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    server.run()


def start_dashboard_server(
    host: str = "0.0.0.0",
    port: int = 8765,
    metrics_file: str = "logs/training/live_metrics.json",
    background: bool = True,
) -> Optional[threading.Thread]:
    """
    Start the dashboard server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
        metrics_file: Path to live_metrics.json
        background: If True, run in background thread (daemon)
    
    Returns:
        Thread object if background=True, else None (blocking)
    """
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
    print("  🚀 PROPFIRM PPO TRAINING DASHBOARD v2.0")
    print("=" * 70)
    print(f"  📊 Open http://localhost:{port} in your browser")
    print(f"  📁 Reading metrics from: {metrics_file}")
    print("  ✨ Features: Skill Assessment, Composite Scoring, Recovery Protocols")
    print("=" * 70)
    print()
    
    if background:
        _server_thread = threading.Thread(
            target=_run_server,
            args=(host, port),
            daemon=True,
            name="DashboardServer",
        )
        _server_thread.start()
        _server_running = True
        time.sleep(0.5)
        return _server_thread
    else:
        _run_server(host, port)
        return None


def stop_dashboard_server():
    """Stop the dashboard server."""
    global _server_running
    _server_running = False
    logger.info("Dashboard server stopping...")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PropFirm PPO Training Dashboard Server v2.0",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py                              # Default settings
  python server.py --port 8080                  # Custom port
  python server.py --metrics-file custom.json   # Custom metrics file
        """
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on")
    parser.add_argument(
        "--metrics-file",
        default="logs/training/live_metrics.json",
        help="Path to live_metrics.json"
    )
    
    args = parser.parse_args()
    
    start_dashboard_server(
        host=args.host,
        port=args.port,
        metrics_file=args.metrics_file,
        background=False,
    )


if __name__ == "__main__":
    main()