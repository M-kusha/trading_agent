#!/usr/bin/env python3
"""
PropFirm PPO Training Dashboard Server
======================================

Real-time dashboard server for monitoring PPO training progress.
Reads metrics from logs/training/live_metrics.json written by VecEpisodeTradingCallback.

Features:
- WebSocket streaming for real-time updates
- REST API for snapshot data
- File watching for automatic updates
- Graceful handling of missing/corrupted data
- Thread-safe operation for embedding in training process

Usage:
    Standalone: python server.py [--port 8765] [--metrics-file path/to/live_metrics.json]
    Embedded:   from traindashboard.server import start_dashboard_server
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

# Web framework - conditional import with type stubs for Pylance
WEB_AVAILABLE = False
if TYPE_CHECKING:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
    import uvicorn

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.staticfiles import StaticFiles
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
    update_interval: float = 0.01  # seconds between WebSocket updates
    file_poll_interval: float = 0.01  # seconds between file checks
    max_history_points: int = 500  # Max data points to keep in memory for charts


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MetricThresholds:
    """Thresholds for color-coding metrics."""
    # Win rate thresholds
    win_rate_good: float = 55.0
    win_rate_ok: float = 45.0
    
    # Drawdown thresholds (lower is better)
    drawdown_good: float = 3.0
    drawdown_ok: float = 6.0
    
    # Explained variance thresholds
    ev_good: float = 0.5
    ev_ok: float = 0.2
    
    # Entropy thresholds (higher is generally better during training)
    entropy_good_min: float = -9.0
    entropy_good_max: float = -5.0
    
    # KL divergence thresholds (lower is more stable)
    kl_good: float = 0.015
    kl_ok: float = 0.025
    
    # Clip fraction thresholds
    clip_good_min: float = 0.05
    clip_good_max: float = 0.20
    
    # R-multiple thresholds
    r_mult_good: float = 0.5
    r_mult_ok: float = 0.0
    
    # Profit factor thresholds
    pf_good: float = 1.5
    pf_ok: float = 1.0
    
    # Entry quality thresholds
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
    Maintains history for trend analysis.
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
        except:
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
            # Check if file exists and has been modified
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
                # Race condition during file write - expected, use cached data
                if self._last_data:
                    return self._last_data
                logger.debug(f"JSON decode error (file being written): {e}")
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
            "risk": {},
            "history": self._history,
            "recent_rewards": [],
            "recent_pnls": [],
            "exit_distribution": {},
        }
    
    def _process_metrics(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw metrics into dashboard format with status colors."""
        
        # Helper to get nested or flat values (supports new nested format and legacy flat format)
        def get_nested(section: str, key: str, default: Any = 0) -> Any:
            """Get value from nested structure or fall back to flat key."""
            if section in raw and isinstance(raw[section], dict):
                return raw[section].get(key, raw.get(key, default))
            return raw.get(key, default)
        
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
        mean_pnl = self._safe_float(raw.get("mean_pnl", 0))
        total_pnl = self._safe_float(learning_section.get("total_pnl", raw.get("total_pnl", 0)))
        
        # PPO diagnostics - read from nested learning section first, then flat
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
        
        # Update history
        self._append_history("entropy", entropy)
        self._append_history("explained_variance", explained_variance)
        self._append_history("kl_divergence", approx_kl)
        
        learning = {
            "mean_reward": mean_reward,
            "mean_reward_status": get_status_color(mean_reward, 0.5, 0, True),
            "total_pnl": total_pnl,
            "total_pnl_status": get_status_color(total_pnl, 500, 0, True),
            "mean_pnl": mean_pnl,
            "mean_pnl_status": get_status_color(mean_pnl, 50, 0, True),
            
            # PPO Diagnostics with status
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
        # Win rate: nested is already percentage (0-100), flat was 0-1
        mean_win_rate_raw = trading_section.get("mean_win_rate", raw.get("mean_win_rate", 0))
        # If value is less than 1, it's a ratio (0-1), convert to percentage
        mean_win_rate = self._safe_float(mean_win_rate_raw)
        if mean_win_rate < 1 and mean_win_rate > 0:
            mean_win_rate = mean_win_rate * 100
        
        # Drawdown: nested is already percentage, flat was 0-1
        max_drawdown_raw = trading_section.get("max_drawdown", raw.get("max_drawdown", 0))
        max_drawdown = self._safe_float(max_drawdown_raw)
        if max_drawdown < 1 and max_drawdown > 0:
            max_drawdown = max_drawdown * 100
            
        mean_trades = self._safe_float(trading_section.get("mean_trades", raw.get("mean_trades", 0)))
        total_trades = self._safe_int(trading_section.get("total_trades", raw.get("total_trades", 0)))
        
        # Update history
        self._append_history("win_rates", mean_win_rate)
        self._append_history("drawdowns", max_drawdown)
        self._append_history("trades_per_episode", mean_trades)
        
        trading = {
            "mean_win_rate": mean_win_rate,
            "mean_win_rate_status": get_status_color(mean_win_rate, THRESHOLDS.win_rate_good, THRESHOLDS.win_rate_ok, True),
            "max_drawdown": max_drawdown,
            "max_drawdown_status": get_status_color(max_drawdown, THRESHOLDS.drawdown_good, THRESHOLDS.drawdown_ok, False),
            "mean_trades": mean_trades,
            "mean_trades_status": get_range_status(mean_trades, 3, 15),  # 3-15 trades per episode is healthy
            "total_trades": total_trades,
        }
        
        # ─────────────────────────────────────────────────────────────
        # TRADE QUALITY METRICS
        # ─────────────────────────────────────────────────────────────
        quality_section = raw.get("quality", {})
        mean_r_multiple = self._safe_float(quality_section.get("mean_r_multiple", raw.get("mean_r_multiple", 0)))
        mean_profit_factor = self._safe_float(quality_section.get("mean_profit_factor", raw.get("mean_profit_factor", 0)))
        mean_mae = self._safe_float(raw.get("mean_mae", 0))
        mean_mfe = self._safe_float(raw.get("mean_mfe", 0))
        mean_bars_held = self._safe_float(raw.get("mean_bars_held", 0))
        mean_entry_quality = self._safe_float(quality_section.get("mean_entry_quality", raw.get("mean_entry_quality", 0.5)))
        max_consecutive_wins = self._safe_int(raw.get("max_consecutive_wins", 0))
        max_consecutive_losses = self._safe_int(raw.get("max_consecutive_losses", 0))
        
        # Update history
        self._append_history("r_multiples", mean_r_multiple)
        
        quality = {
            "mean_r_multiple": mean_r_multiple,
            "mean_r_multiple_status": get_status_color(mean_r_multiple, THRESHOLDS.r_mult_good, THRESHOLDS.r_mult_ok, True),
            "mean_profit_factor": mean_profit_factor,
            "mean_profit_factor_status": get_status_color(mean_profit_factor, THRESHOLDS.pf_good, THRESHOLDS.pf_ok, True),
            "mean_mae": mean_mae,
            "mean_mfe": mean_mfe,
            "mfe_mae_ratio": mean_mfe / max(abs(mean_mae), 1) if mean_mae != 0 else 0,
            "mean_bars_held": mean_bars_held,
            "mean_bars_held_status": get_range_status(mean_bars_held, 4, 20),  # 4-20 bars is reasonable
            "mean_entry_quality": mean_entry_quality,
            "mean_entry_quality_status": get_status_color(mean_entry_quality, THRESHOLDS.eq_good, THRESHOLDS.eq_ok, True),
            "max_consecutive_wins": max_consecutive_wins,
            "max_consecutive_losses": max_consecutive_losses,
            "max_consecutive_losses_status": get_status_color(max_consecutive_losses, 2, 3, False),
        }
        
        # ─────────────────────────────────────────────────────────────
        # EXIT REASON DISTRIBUTION
        # ─────────────────────────────────────────────────────────────
        # Try nested exit_stats.distribution first, then flat exit_reason_distribution
        exit_stats_section = raw.get("exit_stats", {})
        exit_distribution = self._safe_dict(
            exit_stats_section.get("distribution", raw.get("exit_reason_distribution", {}))
        )
        
        # Classify exit reasons
        good_exits = ["trailing_stop", "agent_close"]
        neutral_exits = ["time_decay", "hard_close", "weekend_flatten"]
        bad_exits = ["hard_stop", "emergency_close", "risk_liquidation"]
        
        exit_stats = {
            "distribution": exit_distribution,
            "good_count": sum(exit_distribution.get(e, 0) for e in good_exits),
            "neutral_count": sum(exit_distribution.get(e, 0) for e in neutral_exits),
            "bad_count": sum(exit_distribution.get(e, 0) for e in bad_exits),
        }
        total_exits = exit_stats["good_count"] + exit_stats["neutral_count"] + exit_stats["bad_count"]
        exit_stats["good_pct"] = (exit_stats["good_count"] / total_exits * 100) if total_exits > 0 else 0
        exit_stats["bad_pct"] = (exit_stats["bad_count"] / total_exits * 100) if total_exits > 0 else 0
        
        # ─────────────────────────────────────────────────────────────
        # RECENT DATA FOR CHARTS
        # ─────────────────────────────────────────────────────────────
        recent_rewards = self._safe_list(raw.get("recent_rewards", []))[-100:]
        recent_pnls = self._safe_list(raw.get("recent_pnls", []))[-100:]
        recent_win_rates = self._safe_list(raw.get("recent_win_rates", []))[-100:]
        recent_drawdowns = self._safe_list(raw.get("recent_drawdowns", []))[-100:]
        recent_trades_list = self._safe_list(raw.get("recent_trades", []))[-100:]
        recent_r_multiples = self._safe_list(raw.get("recent_r_multiples", []))[-100:]
        recent_entry_quality = self._safe_list(raw.get("recent_entry_quality", []))[-100:]
        recent_bars_held = self._safe_list(raw.get("recent_bars_held", []))[-100:]
        
        # Update cumulative history
        for r in recent_rewards[-10:]:  # Add last 10 to history
            self._append_history("rewards", self._safe_float(r))
        for p in recent_pnls[-10:]:
            self._append_history("pnls", self._safe_float(p))
        
        # ─────────────────────────────────────────────────────────────
        # ASSEMBLE FINAL PAYLOAD
        # ─────────────────────────────────────────────────────────────
        
        # Handle recent_win_rates and recent_drawdowns conversion
        # If values are already percentage (>1), don't multiply
        def maybe_to_pct(values: List[float]) -> List[float]:
            if not values:
                return []
            # If max value > 1, assume already percentage
            max_val = max(abs(v) for v in values) if values else 0
            if max_val > 1:
                return values
            return [v * 100 for v in values]
        
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
            
            # Curriculum data - passthrough from training callback
            "curriculum_stage": raw.get("curriculum_stage", "N/A"),
            "curriculum_stage_idx": raw.get("curriculum_stage_idx", 0),
            "curriculum_progress": raw.get("curriculum_progress", {}),
            "curriculum_detail": raw.get("curriculum_detail", {}),
            "stage_history": raw.get("stage_history", []),
            
            # Chart data
            "recent_rewards": recent_rewards,
            "recent_pnls": recent_pnls,
            "recent_win_rates": maybe_to_pct(recent_win_rates),
            "recent_drawdowns": maybe_to_pct(recent_drawdowns),
            "recent_trades": recent_trades_list,
            "recent_r_multiples": recent_r_multiples,
            "recent_entry_quality": recent_entry_quality,
            "recent_bars_held": recent_bars_held,
            
            # History for sparklines
            "history": {k: v[-50:] for k, v in self._history.items()},  # Last 50 for sparklines
        }
    
    def _estimate_eta(self, current: int, total: int, raw: Dict) -> float:
        """Estimate time remaining in seconds."""
        fps = self._safe_float(raw.get("fps", 0))
        if fps <= 0 or current <= 0:
            return -1
        remaining = total - current
        return remaining / fps


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI APPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

if WEB_AVAILABLE:
    app = FastAPI(
        title="PropFirm PPO Training Dashboard",
        description="Real-time monitoring for PPO trading agent training",
        version="2.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global state
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
            "training_active": metrics.get("status") == "active",
            "metrics_file": str(_config.metrics_file),
            "connected_clients": len(_connected_clients),
        }
    
    
    @app.get("/api/config")
    async def get_config():
        """Get dashboard configuration."""
        return asdict(_config)
    
    
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
    
    # Update config
    _config = DashboardConfig(host=host, port=port, metrics_file=metrics_file)
    _metrics_reader = MetricsReader(metrics_file)
    
    print()
    print("=" * 70)
    print("  🚀 PROPFIRM PPO TRAINING DASHBOARD")
    print("=" * 70)
    print(f"  📊 Open http://localhost:{port} in your browser")
    print(f"  📁 Reading metrics from: {metrics_file}")
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
        time.sleep(0.5)  # Give server time to start
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
        description="PropFirm PPO Training Dashboard Server",
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
        background=False,  # Run blocking in CLI mode
    )


if __name__ == "__main__":
    main()