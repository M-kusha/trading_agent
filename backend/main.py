#!/usr/bin/env python3
"""
Enhanced AI Trading System Backend - Complete Version
FastAPI server with comprehensive module integration and enhanced training metrics
Production-ready with full monitoring and control capabilities
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
import uuid
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast
import glob

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect,UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import MetaTrader5 as _mt5
import yaml
import re
import websockets

# Help Pylance with third-party modules that lack type stubs
mt5: Any = cast(Any, _mt5)

# Import modules to ensure they are registered with the ModuleOrchestrator
# This is critical for the @module decorator to run and register modules
from modules.market.market_module import UnifiedMarketModule  # Provides market_context, prices, price_data, step_idx

# Fix Windows encoding issues
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("TradingDashboard")


def _bootstrap_debug_logging() -> None:
    """Force DEBUG verbosity globally and for SmartInfoBus (safe, non-fatal)."""
    try:
        # Root + common libraries
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        for h in root_logger.handlers:
            try:
                h.setLevel(logging.DEBUG)
            except Exception:
                pass

        # Our app logger
        logger.setLevel(logging.DEBUG)

        # Uvicorn loggers if present
        for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
            try:
                _l = logging.getLogger(name)
                _l.setLevel(logging.DEBUG)
            except Exception:
                pass

        # SmartInfoBus singleton (if already importable/initialized)
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
            # Update config and internal logger
            try:
                # Make the bus more tolerant to transient module errors by
                # raising the circuit breaker threshold and shortening recovery.
                bus.config.update({
                    "debug_mode": True,
                    "log_level": "DEBUG",
                    "circuit_breaker_threshold": 5,
                    "recovery_time_seconds": 30,
                })
            except Exception:
                pass
            try:
                # Be explicit about I/O verbosity
                setattr(bus, "_verbose_io", True)
            except Exception:
                pass
            try:
                # Elevate RotatingLogger to DEBUG
                if hasattr(bus, "logger") and hasattr(bus.logger, "set_level"):
                    bus.logger.set_level("DEBUG")
            except Exception:
                pass
        except Exception:
            # Defer silently if not available
            pass

        # Elevate audit/logging subsystem if available
        try:
            from modules.utils.audit_utils import get_audit_system  # type: ignore
            audit = get_audit_system()
            if hasattr(audit, "audit_logger") and hasattr(audit.audit_logger, "set_level"):
                audit.audit_logger.set_level("DEBUG")
            if hasattr(audit, "operator_logger") and hasattr(audit.operator_logger, "set_level"):
                audit.operator_logger.set_level("DEBUG")
        except Exception:
            pass

        logger.debug("[BOOT] Debug logging bootstrap applied (global + SmartInfoBus)")
    except Exception:
        # Never fail startup on logging tweaks
        try:
            logger.warning("[BOOT] Failed to apply debug bootstrap; continuing with defaults")
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════
# Cross-Process InfoBus Reader - Reads from training subprocess's persisted data
# ══════════════════════════════════════════════════════════════════════

INFOBUS_PERSISTENCE_FILE = Path("state/infobus_data.json")

def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization.
    Replaces inf, -inf, NaN with None or 0.0 to avoid JSON encoding errors.
    Converts deque, set, frozenset to lists.
    """
    import math
    from collections import deque
    
    if obj is None:
        return None
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0  # Replace inf/nan with 0.0
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (deque, set, frozenset)):
        # Convert deque, set, frozenset to list
        return [sanitize_for_json(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        # Handle objects with __dict__
        try:
            return sanitize_for_json(vars(obj))
        except Exception:
            return str(obj)
    else:
        return obj


# ═══════════════════════════════════════════════════════════════════════
# Cached InfoBus Persistence Reader (prevents file I/O storms)
# ═══════════════════════════════════════════════════════════════════════

_persisted_cache: Dict[str, Any] = {}
_persisted_cache_time: float = 0.0
_CACHE_TTL_SECONDS: float = 0.5  # Refresh cache every 500ms max


def _refresh_persisted_cache() -> Dict[str, Any]:
    """Refresh the persisted bus cache if stale."""
    global _persisted_cache, _persisted_cache_time
    
    now = time.time()
    if now - _persisted_cache_time < _CACHE_TTL_SECONDS and _persisted_cache:
        return _persisted_cache
    
    try:
        if not INFOBUS_PERSISTENCE_FILE.exists():
            _persisted_cache = {}
            _persisted_cache_time = now
            return _persisted_cache
        
        with open(INFOBUS_PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract values from the persisted format
        _persisted_cache = {k: v.get('value') if isinstance(v, dict) else v for k, v in data.items()}
        _persisted_cache_time = now
        return _persisted_cache
    except Exception as e:
        # On error, keep old cache and update timestamp to avoid repeated failures
        _persisted_cache_time = now
        logger.debug(f"[PERSISTENCE] Cache refresh failed: {e}")
        return _persisted_cache


def get_persisted_bus_value(key: str, default: Any = None) -> Any:
    """Read a value from the cached persisted InfoBus data (written by training subprocess)."""
    try:
        cache = _refresh_persisted_cache()
        return cache.get(key, default)
    except Exception as e:
        logger.warning(f"[PERSISTENCE] Failed to read key '{key}': {e}")
        return default

def get_all_persisted_bus_values() -> Dict[str, Any]:
    """Read all values from the cached persisted InfoBus data."""
    try:
        return _refresh_persisted_cache().copy()
    except Exception as e:
        logger.warning(f"[PERSISTENCE] Failed to read persisted data: {e}")
        return {}

def get_bus_value_with_fallback(key: str, module: str, default: Any = None) -> Any:
    """Try InfoBus first, then fall back to persisted file."""
    try:
        from modules.utils.info_bus import InfoBusManager
        bus = InfoBusManager.get_instance()
        value = bus.get(key, module, default=None)
        if value is not None:
            return value
    except Exception:
        pass
    # Fall back to persisted file
    return get_persisted_bus_value(key, default)


# Initialize FastAPI
app = FastAPI(
    title="AI Trading Dashboard API",
    version="3.1.0",
    description="Production-ready AI trading system with enhanced training metrics"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Data Models
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class LoginRequest(BaseModel):
    login: int
    password: str
    server: str = "MetaQuotes-Demo"

class PPOTrainingConfig(BaseModel):
    """Enhanced PPO training configuration with mode selection"""
    mode: str = Field(default="offline", description="Training mode: offline or online")
    timesteps: int = Field(default=100000, ge=1000, le=10000000)
    learning_rate: float = Field(default=3e-4, gt=0, le=1)
    batch_size: int = Field(default=64, ge=16, le=512)
    n_epochs: int = Field(default=10, ge=1, le=50)
    gamma: float = Field(default=0.99, ge=0.9, le=0.999)
    n_steps: int = Field(default=2048, ge=128, le=8192)
    clip_range: float = Field(default=0.2, gt=0, le=1)
    ent_coef: float = Field(default=0.01, ge=0, le=1)
    vf_coef: float = Field(default=0.5, ge=0, le=1)
    max_grad_norm: float = Field(default=0.5, gt=0, le=10)
    target_kl: float = Field(default=0.01, gt=0, le=1)
    checkpoint_freq: int = Field(default=10000, ge=1000, le=100000)
    eval_freq: int = Field(default=5000, ge=1000, le=50000)
    num_envs: int = Field(default=1, ge=1, le=8)
    data_dir: str = Field(default="data/processed", description="Directory with CSV files for offline mode")
    initial_balance: float = Field(default=100000.0, gt=0, description="Account balance - loaded from risk_policy.yaml")
    pretrained_model: Optional[str] = Field(default=None, description="Path to pretrained model")
    auto_pretrained: bool = Field(default=False, description="Auto-load latest model if available")
    debug: bool = False

class LiveTradingConfig(BaseModel):
    """Live trading configuration"""
    instruments: List[str] = Field(default=["EURUSD", "XAUUSD"])
    timeframes: List[str] = Field(default=["H1", "H4", "D1"])
    update_interval: int = Field(default=5, ge=1, le=60)
    max_position_size: float = Field(default=0.1, gt=0, le=1)
    max_total_exposure: float = Field(default=0.3, gt=0, le=1)
    min_trade_interval: int = Field(default=60, ge=10, le=3600)
    use_trailing_stop: bool = True
    emergency_drawdown_limit: float = Field(default=0.25, gt=0, le=0.5)
    debug: bool = False

class SystemState(BaseModel):
    """Comprehensive system state"""
    status: str
    mt5_connected: bool
    model_loaded: bool
    active_positions: int
    total_exposure: float
    current_balance: float
    daily_pnl: float
    risk_level: str
    last_update: str
    uptime: str
    errors_count: int
    warnings_count: int

class ModuleStatus(BaseModel):
    """Individual module status"""
    name: str
    enabled: bool
    status: str
    last_update: str
    metrics: Dict[str, Any]
    errors: List[str]

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Global State Management
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class EnhancedTradingSystemState:
    """Advanced state management with comprehensive module tracking"""

    # Explicit attribute types to satisfy static type checkers
    trading_task: Optional[asyncio.Task[Any]]
    monitoring_tasks: List[asyncio.Task[Any]]
    broadcast_lock: Optional[asyncio.Lock]

    live_env: Optional[Any]
    model: Optional[Any]
    last_trade_time: Dict[str, float]
    trading_config: Optional['LiveTradingConfig']
    tensorboard_process: Optional[Any]

    websocket_connections: List[WebSocket]
    training_websocket_connections: List[Any]
    module_states: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]

    def __init__(self):
        self.startup_time = datetime.now()

        # Core system state
        self.system_status = "IDLE"
        self.mt5_connected = False
        self.model_loaded = False
        self.current_session_id = str(uuid.uuid4())

        # Process management
        self.trading_task = None
        self.monitoring_tasks = []

        # Trading state
        self.live_env = None
        self.model = None
        self.last_trade_time = {}
        self.trading_config = None
        self.tensorboard_process = None

        # Performance tracking
        self.performance_metrics = {
            "session_start_time": self.startup_time.isoformat(),
            "start_balance": 0.0,
            "current_balance": 0.0,
            "peak_balance": 0.0,
            "daily_pnl": 0.0,
            "total_pnl": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sortino_ratio": 0.0,
            "trades_today": 0,
            "last_trade_time": None,
            "avg_trade_duration": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "consecutive_wins": 0,
            "consecutive_losses": 0,
            "max_consecutive_wins": 0,
            "max_consecutive_losses": 0,
        }
        
        # Initialize module states - will be populated from real modules
        self.module_states = {}

        # Populate modules from orchestrator if present; else optionally merge from registry
        try:
            if self._sync_modules_from_orchestrator():
                logger.info("[BOOT] Synced modules from Orchestrator")
            else:
                try:
                    self._merge_registry_modules('config/module_registry.yaml')
                except Exception as e:
                    logger.warning(f"[BOOT] Failed to merge registry modules: {e}")
        except Exception as e:
            logger.warning(f"[BOOT] Module population warning: {e}")
        
        # WebSocket connections
        self.websocket_connections = []
        self.training_websocket_connections = []
        # Broadcast coordination (initialized on startup when loop is available)
        self.broadcast_lock: Optional[asyncio.Lock] = None
        
        # Error and warning tracking
        self.errors = []
        self.warnings = []
        self.alerts = []
        
        # System metrics
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "active_connections": 0,
            "requests_per_minute": 0,
            "last_health_check": datetime.now().isoformat(),
    }
        
    def get_uptime(self) -> str:
        """Get system uptime"""
        uptime = datetime.now() - self.startup_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        elif hours > 0:
            return f"{hours}h {minutes}m"
        else:
            return f"{minutes}m {seconds}s"

    def _merge_registry_modules(self, registry_path: str) -> None:
        """Load module names from YAML registry and ensure placeholder state exists for each.

        Keeps existing modules intact; adds missing entries with sensible defaults so the
        frontend can display the full set (e.g., 56 modules) immediately.
        """
        try:
            if not os.path.exists(registry_path):
                return
            with open(registry_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            mods = data.get('modules', {}) or {}
            now = datetime.now().isoformat()
            for name in mods.keys():
                if name in self.module_states:
                    continue
                # Derive a generic status from category if available
                cat = (mods.get(name, {}) or {}).get('category', '')
                status = 'monitoring' if cat in ('risk', 'auditing') else ('analyzing' if cat in ('features','strategy','voting','models','simulation') else 'idle')
                self.module_states[name] = {
                    "enabled": True,
                    "status": status.upper() if isinstance(status, str) else 'IDLE',
                    "category": (cat or '').lower() or 'other',
                    "last_update": now,
                    "errors": [],
                }
        except Exception as e:
            raise

    def _sync_modules_from_orchestrator(self) -> bool:
        """Sync module_states from the live ModuleOrchestrator (no placeholders).

        Returns True if successfully synced; False if orchestrator not available.
        """
        try:
            from modules.core.module_system import ModuleOrchestrator  # type: ignore
        except Exception:
            return False

        try:
            orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()  # type: ignore[attr-defined]
        except Exception:
            return False

        try:
            mods = getattr(orch, 'modules', {}) or {}
            meta = getattr(orch, 'metadata', {}) or {}
            perf = getattr(orch, 'module_performance', {}) or {}
            bus = getattr(orch, 'smart_bus', None)
            now = datetime.now().isoformat()

            new_states: Dict[str, Dict[str, Any]] = {}
            for name in mods.keys():
                m = meta.get(name)
                cat = ''
                try:
                    cat = getattr(m, 'category', '') if m is not None else ''
                except Exception:
                    cat = ''
                enabled = True
                try:
                    if bus is not None and hasattr(bus, 'is_module_enabled'):
                        enabled = bool(bus.is_module_enabled(name))  # type: ignore[attr-defined]
                except Exception:
                    pass
                p = perf.get(name, {}) if isinstance(perf, dict) else {}
                status = 'idle'
                try:
                    # Heuristic: if recent avg_time_ms exists or success_rate shown, mark monitoring/active
                    avg_ms = float(p.get('avg_time_ms', 0.0) or 0.0)
                    err_rate = float(p.get('error_rate', 0.0) or 0.0)
                    if avg_ms > 0.0:
                        status = 'monitoring' if err_rate < 0.5 else 'error'
                except Exception:
                    pass

                new_states[name] = {
                    'enabled': enabled,
                    'status': status,
                    'category': (cat or 'other').lower(),
                    'last_update': now,
                    'errors': [],
                }

            if new_states:
                self.module_states = new_states
                return True
            return False
        except Exception:
            return False
    
    def add_error(self, error: str, module: str = "system"):
        """Add error with enhanced tracking"""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "error": error,
            "severity": "error",
            "session_id": self.current_session_id,
        }
        self.errors.append(error_entry)
        self.errors = self.errors[-1000:]  # Keep last 1000 errors
        
        # Update module status
        if module in self.module_states:
            if "errors" not in self.module_states[module]:
                self.module_states[module]["errors"] = []
            self.module_states[module]["errors"].append(error)
            self.module_states[module]["errors"] = self.module_states[module]["errors"][-10:]
        
    def add_warning(self, warning: str, module: str = "system"):
        """Add warning with enhanced tracking"""
        warning_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "warning": warning,
            "severity": "warning",
            "session_id": self.current_session_id,
        }
        self.warnings.append(warning_entry)
        self.warnings = self.warnings[-1000:]
        
    def get_category_summary(self) -> Dict[str, Any]:
        """Summarize module counts by category (total/enabled/errors/active)."""
        summary: Dict[str, Dict[str, int]] = {}
        for name, mod in self.module_states.items():
            cat = str(mod.get("category", "unknown")).lower()
            bucket = summary.setdefault(cat, {"total": 0, "enabled": 0, "with_errors": 0, "active": 0})
            bucket["total"] += 1
            if mod.get("enabled", False):
                bucket["enabled"] += 1
            if mod.get("errors"):
                bucket["with_errors"] += 1
            status = str(mod.get("status", "")).lower()
            if status not in ("idle", "unknown", "disabled"):
                bucket["active"] += 1
        return summary
        
    def add_alert(self, alert: str, severity: str = "info", module: str = "system"):
        """Add system alert"""
        alert_entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "alert": alert,
            "severity": severity,
            "session_id": self.current_session_id,
        }
        self.alerts.append(alert_entry)
        self.alerts = self.alerts[-500:]

    def get_training_progress(self) -> Optional[Dict[str, Any]]:
        """Get training progress (stub - training removed)"""
        return None

# Global state instance
state = EnhancedTradingSystemState()

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Training Metrics WebSocket Server
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

class TrainingMetricsServer:
    """WebSocket server - DISABLED FOR LIVE TRADING ONLY"""
    pass

# Global training metrics server - REMOVED FOR LIVE TRADING ONLY
# training_metrics_server = TrainingMetricsServer()  # DISABLED

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MT5 Integration
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def connect_mt5(login: int, password: str, server: str) -> Dict[str, Any]:
    """Enhanced MT5 connection with comprehensive error handling"""
    try:
        logger.info(f"Attempting MT5 connection - Login: {login}, Server: {server}")
        
        # Initialize MT5
        if not mt5.initialize():
            error_code = mt5.last_error()
            error_msg = f"MT5 initialization failed: {error_code}"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}
        
        # Login
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            error_code = mt5.last_error()
            mt5.shutdown()
            error_msg = f"MT5 login failed: {error_code}"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            error_msg = "Failed to retrieve MT5 account information"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}
        
        # Update state
        state.mt5_connected = True
        state.performance_metrics["start_balance"] = account_info.balance
        state.performance_metrics["current_balance"] = account_info.balance
        state.performance_metrics["peak_balance"] = account_info.balance
        
        logger.info(f"MT5 connected successfully - Balance: ${account_info.balance:.2f}")
        
        return {
            "success": True,
            "account": {
                "login": account_info.login,
                "balance": account_info.balance,
                "equity": account_info.equity,
                "margin": account_info.margin,
                "free_margin": account_info.margin_free,
                "currency": account_info.currency,
                "leverage": account_info.leverage,
                "profit": account_info.profit,
                "margin_level": account_info.margin_level,
                "server": server,
                "company": account_info.company,
            }
        }
        
    except Exception as e:
        error_msg = f"MT5 connection error: {str(e)}"
        state.add_error(error_msg, "mt5")
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

def disconnect_mt5():
    """Enhanced MT5 disconnection"""
    try:
        if state.mt5_connected:
            mt5.shutdown()
            state.mt5_connected = False
            logger.info("MT5 disconnected successfully")
        
    except Exception as e:
        error_msg = f"MT5 disconnection error: {str(e)}"
        state.add_error(error_msg, "mt5")
        logger.error(error_msg)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Live Trading System
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def start_live_trading(config: LiveTradingConfig):
    """Enhanced live trading with comprehensive monitoring"""
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")

        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Trading already active")

        # STEP 1: Set environment config on InfoBus BEFORE loading anything
        # This ensures modules that initialize will see the correct mode
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()

            # Create environment config with live mode
            # IMPORTANT: Normalize instrument symbols for module interoperability.
            # - Live connector expects MT5 symbols like 'EURUSD'.
            # - Downstream modules commonly use 'EUR/USD' (and some publish with 'EUR_USD').
            #   Using 'EUR/USD' here ensures PositionManager variants can match both
            #   price_data (published as 'EUR/USD') and voting signals (often 'EUR_USD').
            def _norm_inst(s: str) -> str:
                try:
                    s = str(s)
                    if "/" in s:
                        # Already slash-form
                        return s
                    if "_" in s:
                        # Convert underscore to slash
                        return s.replace("_", "/")
                    if len(s) == 6:
                        return s[:3] + "/" + s[3:]
                except Exception:
                    pass
                return s
            normalized_instruments = [_norm_inst(s) for s in (config.instruments or [])]

            environment_config = {
                "instruments": normalized_instruments,
                "initial_balance": state.performance_metrics["current_balance"],
                "mode": "live",  # CRITICAL: Set to live mode
                "max_steps": 100000,
                "bus_data_active": True,
            }

            # Publish environment_config and execution_mode FIRST
            bus.set("environment_config", environment_config, module="Backend", thesis="live trading environment configuration")
            bus.set("execution_mode", "live", module="Backend", thesis="live trading mode enabled")
            logger.info(f"[LIVE MODE] Set environment config on InfoBus: {environment_config}")
        except Exception as e:
            logger.error(f"Failed to set environment config on InfoBus: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to configure live mode: {e}")

        # STEP 2: Load PPO model
        model_path = "models/ppo_trading_model.zip"
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = ["models/ppo_final_model.zip", "models/modern_ppo_final.zip"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail="PPO model not found")

        logger.info(f"Starting live trading system with model: {model_path}")

        # STEP 3: Import trading components
        from stable_baselines3 import PPO
        from envs.env import EnhancedTradingEnv, TradingConfig
        # Adapt to available connector class name
        try:
            from live.live_connector import LiveDataConnector  # type: ignore
        except Exception:
            from live.live_connector import InfoBusLiveDataConnector as LiveDataConnector  # type: ignore

        # STEP 4: Create live data connector
        connector = LiveDataConnector(
            instruments=config.instruments,
            timeframes=config.timeframes
        )
        connector.connect()

        # STEP 5: Get historical data
        # Support connectors that expose InfoBus-specific method name
        if hasattr(connector, 'get_historical_data'):
            hist_data = connector.get_historical_data(n_bars=1000)  # type: ignore
        elif hasattr(connector, 'get_historical_data_with_infobus'):
            hist_data = connector.get_historical_data_with_infobus(n_bars=1000)  # type: ignore
        else:
            raise HTTPException(status_code=500, detail="Connector does not support historical data retrieval")
        if not hist_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve historical data")

        # STEP 6: Create trading environment
        env_config = TradingConfig(
            initial_balance=state.performance_metrics["current_balance"],
            live_mode=True,
            debug=config.debug,
            max_position_pct=config.max_position_size,
            max_total_exposure=config.max_total_exposure,
        )

        state.live_env = EnhancedTradingEnv(hist_data, env_config)
        state.model = PPO.load(model_path, device="cpu")
        state.model_loaded = True
        state.trading_config = config

        # STEP 7: Start trading loop (will initialize orchestrator inside)
        state.trading_task = asyncio.create_task(
            live_trading_loop(config, connector)
        )

        state.system_status = "TRADING"
        state.add_alert("Live trading started successfully", "success", "trading")
        logger.info("Live trading started successfully")

        # Immediately broadcast so frontend shows TRADING status
        await broadcast_system_state()

        return {"success": True, "message": "Live trading started", "session_id": state.current_session_id}

    except Exception as e:
        error_msg = f"Failed to start live trading: {str(e)}"
        state.add_error(error_msg, "trading")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def live_trading_loop(config: LiveTradingConfig, connector):
    """Enhanced live trading loop with comprehensive monitoring and full module system"""
    try:
        # Help type checker know these are set
        assert state.live_env is not None
        assert state.model is not None
        obs, _ = state.live_env.reset()
        step_count = 0
        last_balance_update = time.time()
        last_health_check = time.time()
        last_state_save = time.time()  # NEW: Track state saving
        STATE_SAVE_INTERVAL = 300  # Save module states every 5 minutes

        # Initialize ModuleOrchestrator for live trading
        # Environment config should already be set by start_live_trading()
        orchestrator = None
        try:
            from modules.core.module_system import ModuleOrchestrator
            orchestrator = ModuleOrchestrator.get_instance()
            # Run synchronous initialize() in executor to avoid blocking event loop
            await asyncio.get_event_loop().run_in_executor(None, orchestrator.initialize)
            logger.info("ModuleOrchestrator initialized for live trading")
        except Exception as e:
            logger.warning(f"Failed to initialize ModuleOrchestrator: {e}")

        logger.info("Live trading loop started with full module system")

        while state.system_status == "TRADING":
            loop_start = time.time()

            try:
                # Update market data
                new_data = connector.get_historical_data(n_bars=1)
                if new_data:
                    update_environment_data(new_data, config)

                # Run ModuleOrchestrator step if available (critical for live trading)
                if orchestrator:
                    try:
                        # Execute all modules (memory, risk, voting, etc.)
                        # Prepare market_data dict from new_data or create minimal dict
                        market_data = new_data if new_data else {}
                        await orchestrator.execute_step(market_data)

                        # Sync module states to backend state
                        state._sync_modules_from_orchestrator()

                    except Exception as e:
                        logger.warning(f"Module execution error: {e}")

                # Get model prediction
                action, _ = state.model.predict(obs, deterministic=True)

                # Execute trading step
                obs, reward, terminated, truncated, info = state.live_env.step(action)
                
                # Update module states from environment info
                update_comprehensive_module_states(info)
                
                # Update performance metrics
                if time.time() - last_balance_update > 30:  # Every 30 seconds
                    update_balance_from_broker()
                    last_balance_update = time.time()
                
                # Health checks
                if time.time() - last_health_check > 60:  # Every minute
                    perform_health_checks()
                    last_health_check = time.time()
                
                # NEW: Periodic state saving for learning persistence
                if orchestrator and time.time() - last_state_save > STATE_SAVE_INTERVAL:
                    try:
                        results = orchestrator.state_manager.save_all_module_states(orchestrator)
                        saved = sum(1 for ok in results.values() if ok)
                        logger.info(f"[SAVE] Periodic state save: {saved}/{len(results)} modules saved")
                        last_state_save = time.time()
                    except Exception as e:
                        logger.warning(f"Periodic state save failed: {e}")
                
                # Emergency checks
                if check_emergency_conditions():
                    logger.warning("Emergency conditions detected, stopping trading")
                    await emergency_stop()
                    break
                
                # Broadcast updates to frontend
                await broadcast_system_state()
                
                step_count += 1
                
                # Sleep to maintain update interval
                loop_time = time.time() - loop_start
                sleep_time = max(0, config.update_interval - loop_time)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Trading loop step error: {str(e)}"
                state.add_error(error_msg, "trading")
                logger.error(error_msg)
                await asyncio.sleep(config.update_interval)

    except Exception as e:
        error_msg = f"Trading loop fatal error: {str(e)}"
        state.add_error(error_msg, "trading")
        logger.error(error_msg)
        state.system_status = "ERROR"
    finally:
        # Reset execution mode to SIM on InfoBus when loop exits
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            bus.set("execution_mode", "sim", module="Backend", thesis="live trading loop ended")
            logger.info("Execution mode reset to SIM on InfoBus (loop cleanup)")
        except Exception as e:
            logger.error(f"Failed to reset execution mode on InfoBus: {e}")

        connector.disconnect()
        logger.info("Live trading loop ended")

def update_environment_data(new_data: Dict, config: LiveTradingConfig):
    """Update environment with new market data"""
    try:
        for inst in config.instruments:
            inst_key = inst[:3] + "/" + inst[3:] if len(inst) == 6 else inst
            for tf in config.timeframes:
                if inst_key in new_data and tf in new_data[inst_key]:
                    if state.live_env is not None and hasattr(state.live_env, 'data') and inst_key in state.live_env.data:
                        state.live_env.data[inst_key][tf] = pd.concat([
                            state.live_env.data[inst_key][tf].iloc[1:],
                            new_data[inst_key][tf].iloc[-1:]
                        ])
                        
    except Exception as e:
        state.add_error(f"Data update error: {str(e)}", "data")

def update_comprehensive_module_states(info: Dict[str, Any]):
    """Enhanced module state updates with comprehensive data extraction"""
    try:
        current_time = datetime.now().isoformat()
        
        # Position Manager updates
        if "position_manager" in info:
            pm_info = info["position_manager"]
            state.module_states["position_manager"].update({
                "status": "active",
                "last_update": current_time,
                "open_positions": pm_info.get("open_positions", {}),
                "total_exposure": pm_info.get("total_exposure", 0.0),
                "position_count": pm_info.get("position_count", 0),
                "avg_holding_time": pm_info.get("avg_holding_time", 0),
                "confidence_scores": pm_info.get("confidence_scores", {}),
                "decision_rationale": pm_info.get("decision_rationale", {}),
            })
        
        # Risk Controller updates
        if "risk" in info:
            risk_info = info["risk"]
            state.module_states["risk_controller"].update({
                "status": "monitoring",
                "last_update": current_time,
                "risk_scale": risk_info.get("risk_scale", 1.0),
                "risk_level": risk_info.get("risk_level", "NORMAL"),
                "volatility": risk_info.get("volatility", {}),
                "var_95": risk_info.get("var_95", 0.0),
                "var_99": risk_info.get("var_99", 0.0),
                "drawdown": risk_info.get("drawdown", 0.0),
                "volatility_ratio": risk_info.get("volatility_ratio", 1.0),
                "risk_budget_used": risk_info.get("risk_budget_used", 0.0),
            })
        
        # Strategy Arbiter updates
        if "votes" in info:
            vote_info = info["votes"]
            state.module_states["strategy_arbiter"].update({
                "status": "voting",
                "last_update": current_time,
                "consensus": vote_info.get("consensus", 0.0),
                "member_votes": vote_info.get("member_votes", {}),
                "member_weights": vote_info.get("weights", []),
                "gate_status": vote_info.get("gate_status", "OPEN"),
                "collusion_score": vote_info.get("collusion_score", 0.0),
                "decision_confidence": vote_info.get("confidence", 0.0),
            })
        
        # Execution Monitor updates
        if "execution" in info:
            exec_info = info["execution"]
            state.module_states["execution_monitor"].update({
                "status": "monitoring",
                "last_update": current_time,
                "slippage": exec_info.get("slippage", 0.0),
                "fill_rate": exec_info.get("fill_rate", 1.0),
                "avg_spread": exec_info.get("avg_spread", 0.0),
                "execution_quality": exec_info.get("quality_score", 1.0),
                "latency_ms": exec_info.get("latency_ms", 0.0),
            })
        
        # Theme Detector updates
        if "themes" in info:
            theme_info = info["themes"]
            state.module_states["theme_detector"].update({
                "status": "analyzing",
                "last_update": current_time,
                "active_themes": theme_info.get("active", []),
                "theme_strengths": theme_info.get("strengths", {}),
                "market_regime": theme_info.get("regime", "NEUTRAL"),
                "regime_confidence": theme_info.get("regime_confidence", 0.0),
            })
        
        # Memory Systems updates
        if "memory" in info:
            memory_info = info["memory"]
            state.module_states["memory_systems"].update({
                "status": "learning",
                "last_update": current_time,
                "mistake_count": memory_info.get("mistakes", 0),
                "playbook_size": memory_info.get("playbook_size", 0),
                "memory_usage": memory_info.get("usage_pct", 0.0),
                "compression_ratio": memory_info.get("compression", 1.0),
                "pattern_matches": memory_info.get("pattern_matches", 0),
            })
        
        # Anomaly Detector updates
        if "anomaly" in info:
            anomaly_info = info["anomaly"]
            state.module_states["anomaly_detector"].update({
                "status": "scanning",
                "last_update": current_time,
                "anomaly_score": anomaly_info.get("score", 0.0),
                "anomalies_detected": anomaly_info.get("detected", []),
                "detection_sensitivity": anomaly_info.get("sensitivity", 0.5),
            })
        
    except Exception as e:
        state.add_error(f"Module state update error: {str(e)}", "system")

def update_balance_from_broker():
    """Update balance from MT5 broker"""
    try:
        if state.mt5_connected:
            account_info = mt5.account_info()
            if account_info:
                old_balance = state.performance_metrics["current_balance"]
                new_balance = account_info.balance
                
                state.performance_metrics["current_balance"] = new_balance
                state.performance_metrics["daily_pnl"] = new_balance - state.performance_metrics["start_balance"]
                state.performance_metrics["total_pnl"] = new_balance - state.performance_metrics["start_balance"]
                
                if new_balance > state.performance_metrics["peak_balance"]:
                    state.performance_metrics["peak_balance"] = new_balance
                
                # Calculate drawdown
                peak = state.performance_metrics["peak_balance"]
                current_dd = (peak - new_balance) / peak if peak > 0 else 0.0
                state.performance_metrics["current_drawdown"] = current_dd
                
                if current_dd > state.performance_metrics["max_drawdown"]:
                    state.performance_metrics["max_drawdown"] = current_dd
                
    except Exception as e:
        state.add_error(f"Balance update error: {str(e)}", "mt5")

def perform_health_checks():
    """Perform comprehensive system health checks"""
    try:
        # Check MT5 connection
        if state.mt5_connected:
            terminal_info = mt5.terminal_info()
            if not terminal_info or not terminal_info.trade_allowed:
                state.add_warning("MT5 trading not allowed", "mt5")
            
            # Check for positions without SL/TP (critical safety check)
            try:
                positions = mt5.positions_get()
                if positions:
                    missing_sl_count = sum(1 for p in positions if p.sl <= 0)
                    missing_tp_count = sum(1 for p in positions if p.tp <= 0)
                    
                    if missing_sl_count > 0:
                        state.add_warning(f"⚠️ {missing_sl_count} positions without Stop Loss!", "risk")
                        logger.warning(f"[RISK] {missing_sl_count} positions without Stop Loss - auto-fixing...")
                        # Auto-fix positions without SL/TP
                        asyncio.create_task(auto_fix_sl_tp())
                    
                    if missing_tp_count > 0:
                        state.add_warning(f"⚠️ {missing_tp_count} positions without Take Profit", "risk")
            except Exception as e:
                logger.error(f"Error checking positions SL/TP: {e}")
        
        # Check model status
        if state.model_loaded and state.model is None:
            state.add_error("Model loaded flag set but model is None", "model")
            state.model_loaded = False
        
        # Check memory usage
        import psutil
        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            state.add_warning(f"High memory usage: {memory_percent:.1f}%", "system")
        
        state.system_metrics.update({
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": memory_percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "last_health_check": datetime.now().isoformat(),
        })
        
    except Exception as e:
        state.add_error(f"Health check error: {str(e)}", "system")


async def auto_fix_sl_tp():
    """Auto-fix positions missing SL/TP in background"""
    try:
        import yaml
        try:
            with open('config/risk_policy.yaml', 'r') as f:
                risk_config = yaml.safe_load(f) or {}
                sl_tp_config = risk_config.get('sl_tp_settings', {})
        except Exception:
            sl_tp_config = {'auto_sl_enabled': True, 'auto_tp_enabled': True}

        if not sl_tp_config.get('fix_missing_sl_tp', True):
            return

        positions = mt5.positions_get()
        if not positions:
            return

        def pips_to_price(symbol: str, pips: float) -> float:
            sym_upper = symbol.upper()
            if 'XAU' in sym_upper or 'GOLD' in sym_upper:
                return pips * 0.01
            elif 'JPY' in sym_upper:
                return pips * 0.01
            else:
                return pips * 0.0001

        fixed_count = 0
        for pos in positions:
            needs_sl = pos.sl <= 0 and sl_tp_config.get('auto_sl_enabled', True)
            needs_tp = pos.tp <= 0 and sl_tp_config.get('auto_tp_enabled', True)
            
            if not needs_sl and not needs_tp:
                continue

            symbol = pos.symbol
            symbol_config = sl_tp_config.get(symbol, sl_tp_config.get('default', {}))
            sl_pips = symbol_config.get('stop_loss_pips', 50)
            tp_pips = symbol_config.get('take_profit_pips', 100)
            
            sl_distance = pips_to_price(symbol, sl_pips)
            tp_distance = pips_to_price(symbol, tp_pips)
            
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info else 5
            
            new_sl = pos.sl
            new_tp = pos.tp
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            
            if is_buy:
                if needs_sl and sl_pips > 0:
                    new_sl = round(pos.price_open - sl_distance, digits)
                if needs_tp and tp_pips > 0:
                    new_tp = round(pos.price_open + tp_distance, digits)
            else:
                if needs_sl and sl_pips > 0:
                    new_sl = round(pos.price_open + sl_distance, digits)
                if needs_tp and tp_pips > 0:
                    new_tp = round(pos.price_open - tp_distance, digits)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl if new_sl > 0 else 0.0,
                "tp": new_tp if new_tp > 0 else 0.0,
            }
            
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                fixed_count += 1
                logger.info(f"[AUTO-SL/TP] Fixed position {pos.ticket}: SL={new_sl:.5f} TP={new_tp:.5f}")

        if fixed_count > 0:
            logger.info(f"[AUTO-SL/TP] Auto-fixed {fixed_count} positions")

    except Exception as e:
        logger.error(f"[AUTO-SL/TP] Error: {e}")

def check_emergency_conditions() -> bool:
    """Enhanced emergency condition checking"""
    try:
        # Check maximum drawdown
        current_dd = state.performance_metrics.get("current_drawdown", 0.0)
        max_dd_limit = state.trading_config.emergency_drawdown_limit if state.trading_config else 0.25
        
        if current_dd > max_dd_limit:
            state.add_alert(f"Emergency: Drawdown {current_dd:.1%} exceeds limit {max_dd_limit:.1%}", "critical", "risk")
            return True
        
        # Check risk controller state
        risk_state = state.module_states.get("risk_controller", {})
        if risk_state.get("freeze_counter", 0) > 10:
            state.add_alert("Emergency: Risk system frozen too long", "critical", "risk")
            return True
        
        # Check correlation risk
        corr_state = state.module_states.get("correlation_controller", {})
        if corr_state.get("max_correlation", 0) > 0.95:
            state.add_alert("Emergency: Extreme correlation detected", "critical", "risk")
            return True
        
        # Check error count
        if len(state.errors) > 100:
            state.add_alert("Emergency: Too many system errors", "critical", "system")
            return True
        
        return False
        
    except Exception as e:
        state.add_error(f"Emergency check error: {str(e)}", "system")
        return False

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Training Management Enhanced
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def start_training(config: PPOTrainingConfig):
    """Start PPO training with enhanced mode selection and metrics"""
    try:
        if state.training_process and state.training_process.poll() is None:
            raise HTTPException(status_code=400, detail="Training already in progress")
        
        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Cannot train while trading is active")
        
        logger.info(f"Starting PPO training in {config.mode.upper()} mode...")
        
        # Build training command with mode selection
        cmd = [
            sys.executable, 
            "train/train_ppo_hybrid.py",
            "--mode", config.mode,  # Pass the mode explicitly
            "--timesteps", str(config.timesteps),
            "--lr", str(config.learning_rate),
            "--batch_size", str(config.batch_size),
            "--n_epochs", str(config.n_epochs),
            "--gamma", str(config.gamma),
            "--n_steps", str(config.n_steps),
            "--clip_range", str(config.clip_range),
            "--ent_coef", str(config.ent_coef),
            "--vf_coef", str(config.vf_coef),
            "--max_grad_norm", str(config.max_grad_norm),
            "--target_kl", str(config.target_kl),
            "--checkpoint_freq", str(config.checkpoint_freq),
            "--eval_freq", str(config.eval_freq),
            "--num_envs", str(config.num_envs),
            "--data_dir", config.data_dir,
            "--balance", str(config.initial_balance),
        ]
        
        # Add pretrained model if specified
        if config.pretrained_model:
            cmd.extend(["--pretrained", config.pretrained_model])
        elif config.auto_pretrained:
            cmd.append("--auto-pretrained")
            
        if config.debug:
            cmd.append("--debug")
        
        # Ensure MT5 is connected for online mode
        if config.mode == "online" and not state.mt5_connected:
            raise HTTPException(
                status_code=400, 
                detail="MT5 must be connected for online training. Please login first."
            )
        
        # Start training process
        state.training_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=os.getcwd(),
            universal_newlines=True,
            bufsize=1
        )
        
        state.system_status = "TRAINING"
        state.training_mode = config.mode
        state.training_start_time = datetime.now()
        state.training_config = config
        state.training_metrics = {}
        state.training_metrics_history = []
        
        state.add_alert(f"PPO training started in {config.mode.upper()} mode", "success", "training")
        logger.info(f"PPO training started with PID: {state.training_process.pid}")
        
        # Start monitoring training process
        asyncio.create_task(monitor_training_process())
        
        return {
            "success": True, 
            "pid": state.training_process.pid,
            "mode": config.mode,
            "config": config.dict(),
            "session_id": state.current_session_id
        }
        
    except Exception as e:
        error_msg = f"Training start error: {str(e)}"
        state.add_error(error_msg, "training")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def monitor_training_process():
    """Enhanced training process monitoring"""
    try:
        if not state.training_process:
            return
        
        # Read training output in real-time
        while state.training_process.poll() is None:
            assert state.training_process.stdout is not None
            output = state.training_process.stdout.readline()
            if output:
                logger.info(f"Training: {output.strip()}")
                
                # Parse special output patterns
                if "Episode reward:" in output:
                    try:
                        reward = float(output.split("Episode reward:")[-1].strip())
                        state.training_metrics["last_episode_reward"] = reward
                    except:
                        pass
                        
            await asyncio.sleep(0.1)
        
        # Process completed
        return_code = state.training_process.poll()
        if return_code == 0:
            state.add_alert("Training completed successfully", "success", "training")
            logger.info("Training completed successfully")
        else:
            error_output = ""
            if state.training_process.stderr is not None:
                error_output = state.training_process.stderr.read()
            state.add_error(f"Training failed with code {return_code}: {error_output}", "training")
            logger.error(f"Training failed: {error_output}")
        
        state.training_mode = None
        state.training_start_time = None
        
        if state.system_status == "TRAINING":
            state.system_status = "IDLE"
        
    except Exception as e:
        state.add_error(f"Training monitoring error: {str(e)}", "training")

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# WebSocket Management Enhanced
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def broadcast_system_state():
    """Broadcast comprehensive system state including training metrics"""
    if not state.websocket_connections:
        logger.debug("[WS] No websocket connections for system_state broadcast")
        return
    
    try:
        logger.debug(f"[WS] Broadcasting system_state to {len(state.websocket_connections)} clients, status={state.system_status}")
        # Optional analytics enrichments from InfoBus
        regime_analytics: Dict[str, Any] = {}
        trading_analytics: Dict[str, Any] = {}
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
            # Regime analytics
            regime_pred = bus.get('regime_prediction', 'BackendAPI', default=None)
            regime_probs = bus.get('regime_probabilities', 'BackendAPI', default={}) or {}
            regime_perf = bus.get('regime_performance', 'BackendAPI', default={}) or {}
            regime_matrix = bus.get('regime_matrix_analysis', 'BackendAPI', default={}) or {}

            # Normalize current_regime to a simple string when possible
            def _norm_regime(val: Any) -> Any:
                try:
                    if isinstance(val, str):
                        return val
                    if isinstance(val, dict):
                        for k in ('label', 'name', 'regime', 'state'):
                            if k in val and isinstance(val[k], str):
                                return val[k]
                    if isinstance(val, (list, tuple)) and val:
                        if isinstance(val[0], str):
                            return val[0]
                        if isinstance(val[0], dict):
                            for k in ('label', 'name', 'regime'):
                                if k in val[0] and isinstance(val[0][k], str):
                                    return val[0][k]
                except Exception:
                    pass
                return val

            regime_analytics = {
                "current_regime": _norm_regime(regime_pred),
                "regime_probabilities": regime_probs,
                "regime_performance": regime_perf,
                "regime_matrix_analysis": regime_matrix,
            }

            # Trading analytics (lightweight)
            trading_analytics = {
                "active_strategy": bus.get('active_strategy', 'BackendAPI', default=None),
                "risk_level": bus.get('risk_level', 'BackendAPI', default=None),
                "auto_mode": bus.get('auto_mode', 'BackendAPI', default=None),
            }
        except Exception:
            pass

        # Include training progress in system state
        system_state = {
            "type": "system_state",
            "data": {
                "status": state.system_status,
                "mt5_connected": state.mt5_connected,
                "model_loaded": state.model_loaded,
                "session_id": state.current_session_id,
                "uptime": state.get_uptime(),
                "performance": state.performance_metrics,
                "modules": state.module_states,
                "alerts": state.alerts[-10:],
                "system_metrics": state.system_metrics,
                "training_progress": state.get_training_progress(),
                "regime_analytics": regime_analytics,
                "trading_analytics": trading_analytics,
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        # Sanitize before sending to prevent numpy type errors
        system_state = sanitize_for_json(system_state)
        
        # Send to all connected clients via common helper (with locking)
        logger.debug(f"[WS] Sending system_state message, size={len(str(system_state))} chars")
        await _send_to_all_websockets(system_state)

    except Exception as e:
        logger.error(f"[WS] Broadcast system_state error: {str(e)}")
        state.add_error(f"Broadcast error: {str(e)}", "websocket")

async def broadcast_mt5_data_update():
    """Broadcast MT5 data updates including live positions"""
    if not state.websocket_connections:
        return

    try:
        # Get fresh MT5 data (filter to EURUSD and XAUUSD only)
        chart_data = None  # Do not push chart data via WS to avoid overwriting HTTP-fetched series
        recent_trades: List[Dict[str, Any]] = []
        symbols: List[Dict[str, Any]] = []
        positions: List[Dict[str, Any]] = []
        account_info: Dict[str, Any] = {}

        # Try to get real MT5 data if connected
        if state.mt5_connected:
            try:
                # Get account info
                acc = mt5.account_info()
                if acc:
                    account_info = {
                        "balance": float(acc.balance),
                        "equity": float(acc.equity),
                        "profit": float(acc.profit),
                        "margin": float(acc.margin),
                        "margin_free": float(acc.margin_free),
                    }
                    # Update state performance metrics with real data
                    state.performance_metrics["current_balance"] = float(acc.balance)
                    if state.performance_metrics.get("start_balance", 0) == 0:
                        state.performance_metrics["start_balance"] = float(acc.balance)
                    state.performance_metrics["total_pnl"] = float(acc.profit)
                
                # Get open positions
                mt5_positions = mt5.positions_get()
                if mt5_positions:
                    for pos in mt5_positions:
                        positions.append({
                            "ticket": pos.ticket,
                            "symbol": pos.symbol,
                            "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                            "volume": float(pos.volume),
                            "price_open": float(pos.price_open),
                            "price_current": float(pos.price_current),
                            "profit": float(pos.profit),
                            "sl": float(pos.sl) if pos.sl else None,
                            "tp": float(pos.tp) if pos.tp else None,
                            "time": datetime.fromtimestamp(pos.time).isoformat(),
                        })
                
                # Get recent trades
                deals = mt5.history_deals_get(datetime.now() - timedelta(days=1), datetime.now())
                if deals:
                    recent_trades = [{
                        "ticket": deal.ticket,
                        "symbol": deal.symbol,
                        "profit": deal.profit,
                        "volume": deal.volume,
                        "time": deal.time,
                        "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL"
                    } for deal in deals[-10:]]  # Last 10 trades

                # Live prices for EURUSD and XAUUSD only
                for sym in ("EURUSD", "XAUUSD"):
                    try:
                        info = mt5.symbol_info(sym)
                        if info is not None:
                            symbols.append({
                                "symbol": getattr(info, 'name', sym) or sym,
                                "description": getattr(info, 'description', sym) or sym,
                                "bid": float(getattr(info, 'bid', 0.0) or 0.0),
                                "ask": float(getattr(info, 'ask', 0.0) or 0.0),
                                "spread": int(getattr(info, 'spread', 0) or 0),
                            })
                        else:
                            symbols.append({
                                "symbol": sym,
                                "description": sym,
                                "bid": 0.0,
                                "ask": 0.0,
                                "spread": 0,
                            })
                    except Exception:
                        symbols.append({
                            "symbol": sym,
                            "description": sym,
                            "bid": 0.0,
                            "ask": 0.0,
                            "spread": 0,
                        })
            except Exception as mt5_error:
                logger.warning(f"Could not get live MT5 data: {mt5_error}")

        payload: Dict[str, Any] = {
            "recentTrades": recent_trades,
            "symbols": symbols,
            "positions": positions,
            "account": account_info,
            "positionCount": len(positions),
            "timestamp": datetime.now().isoformat(),
        }
        if chart_data is not None:
            payload["chartData"] = chart_data

        message = {
            "type": "mt5_data_update",
            "data": payload,
        }

        await _send_to_all_websockets(message)
    except Exception as e:
        logger.error(f"Error broadcasting MT5 data update: {e}")

async def start_real_time_updates():
    """Start periodic real-time updates"""
    logger.info("[WS] Starting real-time update loop")
    while True:
        try:
            # Broadcast system state every 5 seconds
            logger.debug("[WS] Periodic broadcast_system_state call")
            await broadcast_system_state()

            # Broadcast module updates every 10 seconds
            await asyncio.sleep(5)
            await broadcast_modules_update()

            # Broadcast MT5 data every 15 seconds
            await asyncio.sleep(5)
            await broadcast_mt5_data_update()

            # Wait before next cycle
            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")
            await asyncio.sleep(10)  # Wait longer on error

async def broadcast_modules_update():
    """Broadcast only modules data update"""
    if not state.websocket_connections:
        return

    try:
        # Build enriched module list (match /api/modules shape)
        modules_data: List[Dict[str, Any]] = []
        module_registry: Dict[str, Any] = {}

        try:
            with open('config/module_registry.yaml', 'r', encoding='utf-8') as f:
                registry_data = yaml.safe_load(f) or {}
                module_registry = registry_data.get('modules', {}) or {}
        except Exception:
            pass

        # Attempt to get InfoBus for live data
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
        except Exception:
            bus = None

        for name, module in state.module_states.items():
            registry_info = module_registry.get(name, {}) or {}

            live_data: Dict[str, Any] = {}
            if bus:
                try:
                    for key in registry_info.get('provides', []) or []:
                        val = bus.get(key, name, default=None)
                        if val is not None:
                            live_data[key] = val
                except Exception:
                    pass

            health_score = calculate_module_health(module, live_data, registry_info)
            insights = extract_module_insights(name, live_data, module.get("category", "unknown"))
            real_status = determine_real_status(module, live_data, insights)

            modules_data.append({
                "name": name,
                "enabled": module.get("enabled", False),
                "status": real_status,
                "category": (module.get("category") or "other").lower(),
                "last_update": module.get("last_update", datetime.now().isoformat()),
                "file_path": registry_info.get("file_path", ""),
                "provides": registry_info.get("provides", []),
                "requires": registry_info.get("requires", []),
                "live_data": live_data,
                "insights": insights,
                "health_score": health_score,
                "health_status": get_health_status(health_score),
                "metrics": {k: v for k, v in module.items() if k not in ["enabled", "status", "last_update", "errors", "category"]},
                "error_count": len(module.get("errors", [])),
                "errors": module.get("errors", [])[-5:],
                "has_errors": len(module.get("errors", [])) > 0,
                "data_richness": len(live_data),
                "provides_count": len(registry_info.get("provides", [])),
                "requires_count": len(registry_info.get("requires", [])),
            })

        # Categories + stats in the shape the frontend expects
        categories: Dict[str, Dict[str, int]] = {}
        for m in modules_data:
            cat = m["category"]
            bucket = categories.setdefault(cat, {"total": 0, "enabled": 0, "with_data": 0, "with_errors": 0})
            bucket["total"] += 1
            if m["enabled"]:
                bucket["enabled"] += 1
            if m["data_richness"] > 0:
                bucket["with_data"] += 1
            if m["has_errors"]:
                bucket["with_errors"] += 1

        stats = {
            "total": len(modules_data),
            "enabled": sum(1 for m in modules_data if m["enabled"]),
            "withData": sum(1 for m in modules_data if m["data_richness"] > 0),
            "withErrors": sum(1 for m in modules_data if m["has_errors"]),
        }

        message = {
            "type": "modules_update",
            "data": {
                "modules": modules_data,
                "categories": categories,
                "stats": stats,
            }
        }

        # Sanitize before sending to prevent numpy type errors
        message = sanitize_for_json(message)
        await _send_to_all_websockets(message)
    except Exception as e:
        logger.error(f"Error broadcasting modules update: {e}")

async def broadcast_alerts_update():
    """Broadcast only alerts update"""
    if not state.websocket_connections:
        return

    try:
        message = {
            "type": "alerts_update",
            "data": state.alerts[-20:]  # Send last 20 alerts
        }

        await _send_to_all_websockets(message)
    except Exception as e:
        logger.error(f"Error broadcasting alerts update: {e}")

async def broadcast_logs_update(category: str, logs_data: list):
    """Broadcast logs update for specific category"""
    if not state.websocket_connections:
        return

    try:
        message = {
            "type": "logs_update",
            "category": category,
            "data": logs_data
        }

        await _send_to_all_websockets(message)
    except Exception as e:
        logger.error(f"Error broadcasting logs update: {e}")

async def _send_to_all_websockets(message: dict):
    """Helper function to send message to all connected websockets"""
    lock = getattr(state, "broadcast_lock", None)
    
    async def _do_send():
        disconnected = []
        for websocket in list(state.websocket_connections):
            try:
                # Check if websocket is still in a valid state
                if hasattr(websocket, 'client_state'):
                    from starlette.websockets import WebSocketState
                    if websocket.client_state != WebSocketState.CONNECTED:
                        logger.debug(f"[WS] Skipping websocket in state: {websocket.client_state}")
                        disconnected.append(websocket)
                        continue
                await websocket.send_json(message)
            except Exception as e:
                # Only remove on actual connection errors, not transient issues
                error_str = str(e).lower()
                if any(x in error_str for x in ['closed', 'disconnect', 'connection', 'broken pipe']):
                    logger.debug(f"[WS] Removing disconnected websocket: {e}")
                    disconnected.append(websocket)
                else:
                    logger.warning(f"[WS] Error sending to websocket (keeping connection): {e}")
        for ws in disconnected:
            if ws in state.websocket_connections:
                state.websocket_connections.remove(ws)
                logger.debug(f"[WS] Removed websocket, {len(state.websocket_connections)} remaining")
    
    if lock is not None:
        async with lock:  # ensure only one broadcast runs at a time
            await _do_send()
    else:
        # Fallback without lock (startup race)
        await _do_send()

# ═══════════════════════════════════════════════════════════════════
# Emergency Controls# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Emergency Controls
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

async def emergency_stop():
    """Enhanced emergency stop with comprehensive cleanup"""
    try:
        logger.warning("[ALERT] EMERGENCY STOP INITIATED")
        state.add_alert("Emergency stop initiated", "critical", "emergency")
        
        # Close all MT5 positions
        if state.mt5_connected:
            positions = mt5.positions_get()
            if positions:
                logger.info(f"Closing {len(positions)} open positions...")
                
                for position in positions:
                    try:
                        # Determine order type for closing
                        if position.type == mt5.ORDER_TYPE_BUY:
                            order_type = mt5.ORDER_TYPE_SELL
                            price = mt5.symbol_info_tick(position.symbol).bid
                        else:
                            order_type = mt5.ORDER_TYPE_BUY
                            price = mt5.symbol_info_tick(position.symbol).ask
                        
                        request = {
                            "action": mt5.TRADE_ACTION_DEAL,
                            "symbol": position.symbol,
                            "volume": position.volume,
                            "type": order_type,
                            "position": position.ticket,
                            "price": price,
                            "deviation": 20,
                            "magic": 234000,
                            "comment": "Emergency stop",
                        }
                        
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            logger.info(f"[OK] Closed position {position.ticket}")
                        else:
                            logger.error(f"[FAIL] Failed to close position {position.ticket}: {result.comment}")
                            
                    except Exception as e:
                        logger.error(f"Error closing position {position.ticket}: {e}")
        
        # Stop trading loop
        if state.trading_task and not state.trading_task.done():
            state.trading_task.cancel()
            try:
                await state.trading_task
            except asyncio.CancelledError:
                pass

        # NEW: Save all module states before shutdown (preserve learning)
        try:
            from modules.core.module_system import ModuleOrchestrator
            orchestrator = ModuleOrchestrator.get_instance()
            if orchestrator and hasattr(orchestrator, 'state_manager'):
                results = orchestrator.state_manager.save_all_module_states(orchestrator)
                saved = sum(1 for ok in results.values() if ok)
                logger.info(f"[SAVE] Emergency shutdown state save: {saved}/{len(results)} modules saved")
        except Exception as e:
            logger.warning(f"Failed to save module states on emergency stop: {e}")

        # Reset execution mode to SIM on InfoBus
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            bus.set("execution_mode", "sim", module="Backend", thesis="emergency stop, back to simulation")
            logger.info("Execution mode reset to SIM on InfoBus")
        except Exception as e:
            logger.error(f"Failed to reset execution mode on InfoBus: {e}")

        state.system_status = "EMERGENCY_STOPPED"
        state.add_alert("Emergency stop completed", "warning", "emergency")
        logger.warning("[STOP] Emergency stop completed")

        return {"success": True, "message": "Emergency stop executed"}
        
    except Exception as e:
        error_msg = f"Emergency stop error: {str(e)}"
        state.add_error(error_msg, "emergency")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# API Endpoints
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

@app.on_event("startup")
async def startup_event():
    """Enhanced system startup"""
    # Create required directories
    directories = [
        "logs", "logs/training", "logs/risk", "logs/simulation",
        "logs/strategy", "logs/position", "logs/tensorboard",
        "logs/evaluation", "logs/monitoring",
        "models", "models/best", "data", "data/processed",
        "metrics"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    # Force DEBUG-level verbosity for better diagnostics
    _bootstrap_debug_logging()

    # Initialize InfoBus with STANDBY mode (simulation by default)
    # This prevents modules from initializing in wrong mode
    try:
        from modules.utils.info_bus import InfoBusManager
        bus = InfoBusManager.get_instance()

        # Set default simulation mode
        default_env_config = {
            "instruments": [],
            "initial_balance": 100000.0,
            "mode": "sim",  # STANDBY mode = simulation
            "max_steps": 100000,
            "bus_data_active": False,
        }

        bus.set("environment_config", default_env_config, module="Backend", thesis="backend startup - standby mode (sim)")
        bus.set("execution_mode", "sim", module="Backend", thesis="backend startup - standby mode")
        logger.info("[INIT] InfoBus initialized in STANDBY mode (simulation)")
    except Exception as e:
        logger.error(f"Failed to initialize InfoBus in standby mode: {e}")

    # Initialize broadcast lock when event loop is available
    try:
        state.broadcast_lock = asyncio.Lock()
    except Exception:
        state.broadcast_lock = None

    # Start background monitoring tasks
    state.monitoring_tasks.extend([
        asyncio.create_task(periodic_metrics_collector()),
        asyncio.create_task(system_health_monitor()),
        asyncio.create_task(performance_tracker()),
        asyncio.create_task(start_real_time_updates()),  # Start real-time WebSocket updates
    ])

    logger.info("[ROCKET] Enhanced Trading Dashboard Backend Started")
    state.add_alert("System started successfully", "success", "system")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced system shutdown"""
    logger.info("[STOP] Shutting down trading dashboard...")
    
    # Cancel monitoring tasks
    for task in state.monitoring_tasks:
        if not task.done():
            task.cancel()
    
    # Stop trading if active
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        state.trading_task.cancel()
        try:
            await state.trading_task
        except asyncio.CancelledError:
            pass
    
    # Disconnect MT5
    disconnect_mt5()
    
    # Terminate processes
    if state.tensorboard_process and state.tensorboard_process.poll() is None:
        state.tensorboard_process.terminate()
    
    logger.info("[OK] Trading Dashboard Backend Shutdown Complete")

# Background monitoring tasks
async def periodic_metrics_collector():
    """Collect system metrics periodically"""
    while True:
        try:
            await broadcast_system_state()
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(10)

async def system_health_monitor():
    """Monitor system health periodically"""
    while True:
        try:
            perform_health_checks()
            await asyncio.sleep(60)  # Check every minute
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(60)

async def performance_tracker():
    """Track and update performance metrics"""
    while True:
        try:
            # Periodically sync module states from orchestrator (real-time view)
            try:
                state._sync_modules_from_orchestrator()
            except Exception:
                pass
            if state.mt5_connected and state.system_status == "TRADING":
                update_balance_from_broker()
            await asyncio.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logger.error(f"Performance tracking error: {e}")
            await asyncio.sleep(30)

# Authentication endpoints
@app.post("/api/login")
async def login(request: LoginRequest):
    """Enhanced MT5 login"""
    result = connect_mt5(request.login, request.password, request.server)
    if result["success"]:
        await broadcast_system_state()
        return result
    else:
        raise HTTPException(status_code=401, detail=result["error"])

@app.post("/api/logout")
async def logout():
    """Enhanced logout with cleanup"""
    # Stop trading if active
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        state.trading_task.cancel()
        try:
            await state.trading_task
        except asyncio.CancelledError:
            pass
    
    disconnect_mt5()
    state.system_status = "IDLE"
    state.add_alert("User logged out", "info", "auth")
    await broadcast_system_state()
    return {"success": True}

# Live trading endpoints
@app.post("/api/trading/start")
async def trading_start(config: LiveTradingConfig):
    """Start live trading"""
    result = await start_live_trading(config)
    await broadcast_system_state()
    return result

@app.post("/api/trading/stop")
async def trading_stop():
    """Stop live trading"""
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        state.trading_task.cancel()
        try:
            await state.trading_task
        except asyncio.CancelledError:
            pass

        # Reset execution mode to SIM on InfoBus
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            bus.set("execution_mode", "sim", module="Backend", thesis="live trading stopped, back to simulation")
            logger.info("Execution mode reset to SIM on InfoBus")
        except Exception as e:
            logger.error(f"Failed to reset execution mode on InfoBus: {e}")

        state.system_status = "IDLE"
        state.add_alert("Trading stopped by user", "info", "trading")
        await broadcast_system_state()
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="No trading process running")

@app.post("/api/trading/emergency-stop")
async def emergency_stop_endpoint():
    """Emergency stop endpoint"""
    result = await emergency_stop()
    await broadcast_system_state()
    return result

# Enhanced monitoring endpoints
@app.get("/api/status")
async def get_comprehensive_status():
    """Get comprehensive system status"""
    return {
        "system": {
            "status": state.system_status,
            "uptime": state.get_uptime(),
            "session_id": state.current_session_id,
        },
        "connectivity": {
            "mt5_connected": state.mt5_connected,
            "model_loaded": state.model_loaded,
            "active_websockets": len(state.websocket_connections),
        },
        "performance": state.performance_metrics,
        "modules": {name: {
            "enabled": module.get("enabled", False),
            "status": module.get("status", "unknown"),
            "last_update": module.get("last_update", "never"),
        } for name, module in state.module_states.items()},
        "health": {
            "errors_count": len(state.errors),
            "warnings_count": len(state.warnings),
            "alerts_count": len(state.alerts),
            "last_health_check": state.system_metrics.get("last_health_check"),
        },
        "system_metrics": state.system_metrics,
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/data/freshness")
async def get_data_freshness():
    """Get data freshness information to help UI show when data was last updated"""
    try:
        freshness = {
            "training_active": state.training_in_progress,
            "live_active": state.live_trading_active,
            "has_persisted_data": INFOBUS_PERSISTENCE_FILE.exists(),
            "persisted_data_age_seconds": None,
            "data_sources": {},
        }
        
        # Check persisted data age
        if INFOBUS_PERSISTENCE_FILE.exists():
            import os
            mtime = os.path.getmtime(INFOBUS_PERSISTENCE_FILE)
            freshness["persisted_data_age_seconds"] = time.time() - mtime
            freshness["last_data_update"] = datetime.fromtimestamp(mtime).isoformat()
            
            # Read persisted data to check what keys exist
            try:
                with open(INFOBUS_PERSISTENCE_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check key categories
                memory_keys = ['unified_metrics', 'memory_gate', 'playbook_recall', 'mistake_memory', 'neural_memory']
                risk_keys = ['risk_metrics', 'risk_level', 'risk_assessment']
                voting_keys = ['consensus_score', 'committee_votes', 'final_decision']
                
                freshness["data_sources"]["memory"] = {
                    "has_data": any(k in data for k in memory_keys),
                    "keys_present": [k for k in memory_keys if k in data],
                    "last_update": max(
                        (data.get(k, {}).get('timestamp', 0) for k in memory_keys if k in data),
                        default=0
                    )
                }
                freshness["data_sources"]["risk"] = {
                    "has_data": any(k in data for k in risk_keys),
                    "keys_present": [k for k in risk_keys if k in data],
                }
                freshness["data_sources"]["voting"] = {
                    "has_data": any(k in data for k in voting_keys),
                    "keys_present": [k for k in voting_keys if k in data],
                }
                freshness["total_persisted_keys"] = len(data)
            except Exception as e:
                freshness["persisted_data_read_error"] = str(e)
        else:
            freshness["last_data_update"] = None
            
        return {"success": True, **freshness, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/config/system")
async def get_system_configuration():
    """Get system configuration for frontend"""
    try:
        # Load system config from YAML
        config_path = Path(__file__).parent.parent / "config" / "system_config.yaml"
        risk_policy_path = Path(__file__).parent.parent / "config" / "risk_policy.yaml"

        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                system_config = yaml.safe_load(f)
        else:
            system_config = {}
        
        # Load risk policy for initial_balance
        risk_policy = {}
        if risk_policy_path.exists():
            with open(risk_policy_path, 'r', encoding='utf-8') as f:
                risk_policy = yaml.safe_load(f) or {}
        
        # Get initial_balance from risk_policy.yaml
        initial_balance = float(
            risk_policy.get("prop_firm", {}).get("account_size")
            or risk_policy.get("lot_sizing", {}).get("account_balance")
            or 100000.0
        )

        # Extract relevant configuration sections
        response = {
            "trading": {
                "instruments": ["EURUSD", "XAUUSD"],
                "timeframes": ["H1", "H4", "D1"],
                "update_interval": system_config.get("modules", {}).get("MarketDataProvider", {}).get("config", {}).get("update_frequency", 5),
                "max_position_size": 0.1,
                "max_total_exposure": 0.3,
                "min_trade_interval": 60,
                "use_trailing_stop": True,
                "emergency_drawdown_limit": 0.25,
                "debug": False
            },
            "training": {
                "mode": "offline",
                "timesteps": 100000,
                "learning_rate": system_config.get("modules", {}).get("PPOAgent", {}).get("config", {}).get("learning_rate", 3e-4),
                "batch_size": 64,
                "n_epochs": 10,
                "gamma": 0.99,
                "n_steps": 2048,
                "clip_range": system_config.get("modules", {}).get("PPOAgent", {}).get("config", {}).get("clip_eps", 0.2),
                "ent_coef": system_config.get("modules", {}).get("PPOAgent", {}).get("config", {}).get("entropy_coeff", 0.01),
                "vf_coef": system_config.get("modules", {}).get("PPOAgent", {}).get("config", {}).get("value_coeff", 0.5),
                "max_grad_norm": 0.5,
                "target_kl": 0.01,
                "checkpoint_freq": 10000,
                "eval_freq": 5000,
                "num_envs": 1,
                "data_dir": "data/processed",
                "initial_balance": initial_balance,  # From risk_policy.yaml
                "pretrained_model": None,
                "auto_pretrained": False,
                "debug": False
            },
            "mt5": {
                "server": "MetaQuotes-Demo"
            },
            "system": system_config.get("system", {}),
            "monitoring": system_config.get("monitoring", {})
        }

        return response

    except Exception as e:
        logger.error(f"Error loading system configuration: {e}")
        return HTTPException(status_code=500, detail=f"Failed to load configuration: {str(e)}")

@app.get("/api/performance/chart-data")
async def get_performance_chart_data():
    """Get performance chart data for frontend charts"""
    try:
        # Generate realistic performance data based on current metrics
        current_time = datetime.now()
        start_balance = state.performance_metrics.get("start_balance", 10000)
        current_balance = state.performance_metrics.get("current_balance", start_balance)
        total_pnl = state.performance_metrics.get("total_pnl", 0)

        # Generate hourly data points for the last 24 hours
        chart_data = []
        for i in range(24):
            time_point = current_time - timedelta(hours=23-i)
            # Use actual balance data from InfoBus or MT5
            balance = current_balance  # Use current real balance
            pnl = total_pnl  # Use current real PnL

            chart_data.append({
                "time": time_point.strftime("%H:%M"),
                "timestamp": time_point.isoformat(),
                "balance": round(balance, 2),
                "pnl": round(pnl, 2),
                "cumulative_return": round((balance / start_balance - 1) * 100, 2)
            })

        return {
            "success": True,
            "data": chart_data,
            "metadata": {
                "start_balance": start_balance,
                "current_balance": current_balance,
                "total_pnl": total_pnl,
                "timeframe": "24h"
            }
        }

    except Exception as e:
        logger.error(f"Error generating performance chart data: {e}")
        return HTTPException(status_code=500, detail=f"Failed to generate chart data: {str(e)}")

@app.get("/api/trading/symbols")
async def get_trading_symbols():
    """Get available trading symbols from MT5 or configuration"""
    try:
        # Get symbols from MT5 if connected, otherwise from config
        if state.mt5_connected:
            # Try to get symbols from MT5
            try:
                symbols = mt5.symbols_get()
                if symbols:
                    symbol_list = []
                    for symbol in symbols[:20]:  # Limit to top 20
                        symbol_list.append({
                            "symbol": symbol.name,
                            "description": symbol.description,
                            "currency_base": symbol.currency_base,
                            "currency_profit": symbol.currency_profit,
                            "enabled": True
                        })
                    return {"success": True, "symbols": symbol_list}
            except Exception as mt5_error:
                logger.warning(f"Could not get MT5 symbols: {mt5_error}")

        # Fallback to configured symbols
        default_symbols = [
            {"symbol": "EURUSD", "description": "Euro vs US Dollar", "currency_base": "EUR", "currency_profit": "USD", "enabled": True},
            {"symbol": "XAUUSD", "description": "Gold vs US Dollar", "currency_base": "XAU", "currency_profit": "USD", "enabled": True},
            {"symbol": "GBPUSD", "description": "British Pound vs US Dollar", "currency_base": "GBP", "currency_profit": "USD", "enabled": True},
            {"symbol": "USDJPY", "description": "US Dollar vs Japanese Yen", "currency_base": "USD", "currency_profit": "JPY", "enabled": True},
            {"symbol": "AUDUSD", "description": "Australian Dollar vs US Dollar", "currency_base": "AUD", "currency_profit": "USD", "enabled": True}
        ]

        return {"success": True, "symbols": default_symbols}

    except Exception as e:
        logger.error(f"Error getting trading symbols: {e}")
        return HTTPException(status_code=500, detail=f"Failed to get symbols: {str(e)}")

def sanitize_for_json(obj: Any) -> Any:
    """
    Recursively convert numpy types and other non-JSON-serializable objects to Python native types.
    This fixes FastAPI JSON encoder errors with numpy.bool, numpy.int64, NaN/Inf floats, etc.
    """
    from collections import deque
    
    if obj is None:
        return None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        # JSON cannot encode NaN/Inf; treat them as missing values
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return [sanitize_for_json(item) for item in obj.tolist()]
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, (set, frozenset, deque)):
        return [sanitize_for_json(item) for item in obj]
    if hasattr(obj, '__dict__'):
        # Handle objects with __dict__ (pydantic models, etc.)
        try:
            return sanitize_for_json(obj.__dict__)
        except Exception:
            return str(obj)
    return obj

def calculate_module_health(module: Dict[str, Any], live_data: Dict[str, Any], registry_info: Dict[str, Any]) -> int:
    """Calculate module health score (0-100) based on multiple factors"""
    score = 100

    # Enabled status (critical factor)
    if not module.get("enabled", False):
        score -= 50

    # Error rate
    errors = len(module.get("errors", []))
    if errors > 0:
        score -= min(errors * 10, 30)  # Max -30 for errors

    # Data availability
    expected_provides = len(registry_info.get("provides", []))
    actual_provides = len(live_data)
    if expected_provides > 0:
        data_ratio = actual_provides / expected_provides
        score -= int((1.0 - data_ratio) * 20)  # Max -20 for missing data

    # Last update freshness
    last_update = module.get("last_update", "never")
    if last_update == "never":
        score -= 15
    elif isinstance(last_update, str) and "ago" in last_update.lower():
        # Extract time info if available
        if "hour" in last_update or "day" in last_update:
            score -= 10

    # Status quality
    status = module.get("status", "unknown").lower()
    if status in ["unknown", "error", "failed"]:
        score -= 20
    elif status in ["idle", "disabled"]:
        score -= 10

    return max(0, min(100, score))

def get_health_status(score: int) -> str:
    """Convert health score to status string"""
    if score >= 90:
        return "excellent"
    elif score >= 75:
        return "good"
    elif score >= 60:
        return "fair"
    elif score >= 40:
        return "poor"
    else:
        return "critical"

def extract_module_insights(name: str, live_data: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Extract rich insights from module live data based on category"""
    insights = {
        "summary": "No data available",
        "key_metrics": {},
        "alerts": [],
        "performance": {},
        "recommendations": []
    }

    if not live_data:
        return insights

    try:
        # Strategy modules
        if category == "strategy":
            insights["summary"] = extract_strategy_insights(live_data)
            insights["key_metrics"] = {k: v for k, v in live_data.items()
                                     if any(x in k.lower() for x in ["score", "ratio", "performance", "accuracy"])}

        # Risk modules
        elif category == "risk":
            insights["summary"] = extract_risk_insights(live_data)
            insights["key_metrics"] = {k: v for k, v in live_data.items()
                                     if any(x in k.lower() for x in ["risk", "alert", "violation", "threshold"])}

        # Feature modules
        elif category == "features":
            insights["summary"] = extract_feature_insights(live_data)
            insights["key_metrics"] = {k: v for k, v in live_data.items()
                                     if any(x in k.lower() for x in ["feature", "health", "quality", "engine"])}

        # Voting modules
        elif category == "voting":
            insights["summary"] = extract_voting_insights(live_data)
            insights["key_metrics"] = {k: v for k, v in live_data.items()
                                     if any(x in k.lower() for x in ["confidence", "consensus", "vote", "proposal"])}

        # Default extraction
        else:
            insights["summary"] = f"Processing {len(live_data)} data points"
            insights["key_metrics"] = {k: v for k, v in live_data.items() if isinstance(v, (int, float))}

    except Exception as e:
        insights["summary"] = f"Data processing error: {str(e)[:50]}"

    return insights

def extract_strategy_insights(data: Dict[str, Any]) -> str:
    """Extract strategy-specific insights"""
    if "strategy_performance" in data:
        perf = data["strategy_performance"]
        if isinstance(perf, dict):
            return f"Performance tracking: {len(perf)} metrics"
    if "behavior_patterns" in data:
        patterns = data["behavior_patterns"]
        return f"Analyzing {len(patterns) if isinstance(patterns, (list, dict)) else 'behavioral'} patterns"
    if "introspection_metrics" in data:
        return "Deep strategy analysis active"
    return f"Strategy analysis: {len(data)} data streams"

def extract_risk_insights(data: Dict[str, Any]) -> str:
    """Extract risk-specific insights"""
    if "anomaly_score" in data:
        score = data["anomaly_score"]
        if isinstance(score, (int, float)):
            if score > 0.8:
                return f"HIGH RISK: Anomaly score {score:.2f}"
            elif score > 0.5:
                return f"Medium risk: Anomaly score {score:.2f}"
            else:
                return f"Low risk: Anomaly score {score:.2f}"
    if "risk_alerts" in data or "anomaly_alerts" in data:
        return "Active risk monitoring with alerts"
    if "position_duration_risk" in data:
        return "Monitoring position duration risks"
    return f"Risk monitoring: {len(data)} parameters"

def extract_feature_insights(data: Dict[str, Any]) -> str:
    """Extract feature-specific insights"""
    if "feature_health" in data:
        health = data["feature_health"]
        return f"Feature engine health: {health}"
    if "feature_thesis" in data:
        return "Advanced feature thesis analysis active"
    if "advanced_features" in data:
        features = data["advanced_features"]
        return f"Processing {len(features) if isinstance(features, (list, dict)) else 'advanced'} features"
    return f"Feature processing: {len(data)} components"

def extract_voting_insights(data: Dict[str, Any]) -> str:
    """Extract voting-specific insights"""
    if "confidence_bounds" in data:
        return "Confidence analysis with uncertainty bounds"
    if "committee_confidence" in data:
        conf = data["committee_confidence"]
        return f"Committee confidence: {conf}"
    if "consensus" in data:
        consensus = data["consensus"]
        return f"Consensus level: {consensus}"
    return f"Voting analysis: {len(data)} factors"

def determine_real_status(module: Dict[str, Any], live_data: Dict[str, Any], insights: Dict[str, Any]) -> str:
    """Determine real-time status based on live data and insights"""
    base_status = module.get("status", "unknown").lower()

    # If module is disabled, return disabled
    if not module.get("enabled", False):
        return "DISABLED"

    # If we have live data, the module is active
    if live_data:
        category = module.get("category", "").lower()

        # Check for specific indicators in live data
        if any(key in live_data for key in ["error", "failed", "critical"]):
            return "ERROR"
        elif any(key in live_data for key in ["alert", "warning"]):
            return "WARNING"
        elif category == "risk" and any(key in live_data for key in ["anomaly_score", "risk_alerts"]):
            return "MONITORING"
        elif category == "strategy" and any(key in live_data for key in ["analysis", "performance"]):
            return "ANALYZING"
        elif category == "voting" and any(key in live_data for key in ["consensus", "voting"]):
            return "VOTING"
        elif category == "features":
            return "PROCESSING"
        else:
            return "ACTIVE"

    # Fallback to base status, but make it more descriptive
    status_map = {
        "idle": "IDLE",
        "monitoring": "MONITORING",
        "analyzing": "ANALYZING",
        "active": "ACTIVE",
        "voting": "VOTING",
        "learning": "LEARNING",
        "scanning": "SCANNING",
        "unknown": "UNKNOWN"
    }

    return status_map.get(base_status, "UNKNOWN")

@app.get("/api/modules")
async def list_modules():
    """Get comprehensive module states with enhanced data"""
    try:
        modules_data = []

        # Get data from InfoBus for each module
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
        except Exception as e:
            logger.debug(f"InfoBus not available: {e}")
            bus = None

        # Load module registry for additional metadata
        module_registry = {}
        try:
            import yaml
            with open('config/module_registry.yaml', 'r') as f:
                registry_data = yaml.safe_load(f)
                module_registry = registry_data.get('modules', {})
        except Exception as e:
            logger.warning(f"Failed to load module registry: {e}")
            pass

        for name, module in state.module_states.items():
            try:
                # Get registry info for this module
                registry_info = module_registry.get(name, {})

                # Get live data from InfoBus
                live_data = {}
                if bus:
                    try:
                        # Get all data this module provides
                        provides = registry_info.get('provides', [])
                        for key in provides:
                            value = bus.get(key, name, default=None)
                            if value is not None:
                                live_data[key] = value
                    except Exception as e:
                        logger.debug(f"Failed to get live data for {name}: {e}")
                        pass

                # Calculate health score (0-100)
                health_score = calculate_module_health(module, live_data, registry_info)

                # Extract rich insights from live data
                insights = extract_module_insights(name, live_data, module.get("category", "unknown"))

                # Determine real-time status
                real_status = determine_real_status(module, live_data, insights)

                # Enhanced module data
                module_data = {
                    "name": name,
                    "enabled": module.get("enabled", False),
                    "status": real_status,
                    "category": module.get("category", "unknown"),
                    "last_update": module.get("last_update", "never"),
                    "file_path": registry_info.get("file_path", ""),
                    "provides": registry_info.get("provides", []),
                    "requires": registry_info.get("requires", []),
                    "live_data": live_data,
                    "insights": insights,
                    "health_score": health_score,
                    "health_status": get_health_status(health_score),
                    "metrics": {k: v for k, v in module.items()
                              if k not in ["enabled", "status", "last_update", "errors", "category"]},
                    "error_count": len(module.get("errors", [])),
                    "errors": module.get("errors", [])[-5:],  # Last 5 errors
                    "has_errors": len(module.get("errors", [])) > 0,
                    "data_richness": len(live_data),
                    "provides_count": len(registry_info.get("provides", [])),
                    "requires_count": len(registry_info.get("requires", []))
                }
                modules_data.append(module_data)
            except Exception as e:
                logger.error(f"Error processing module {name}: {e}", exc_info=True)
                # Continue with other modules
                continue

        # Sort by category then name
        modules_data.sort(key=lambda x: (x["category"], x["name"]))

        # Calculate category statistics
        categories = {}
        for module in modules_data:
            cat = module["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "enabled": 0, "with_data": 0, "with_errors": 0}
            categories[cat]["total"] += 1
            if module["enabled"]:
                categories[cat]["enabled"] += 1
            if module["data_richness"] > 0:
                categories[cat]["with_data"] += 1
            if module["has_errors"]:
                categories[cat]["with_errors"] += 1

        # Sanitize all data to prevent numpy type JSON serialization errors
        response_data = {
            "modules": modules_data,
            "total_modules": len(modules_data),
            "enabled_modules": sum(1 for m in modules_data if m["enabled"]),
            "categories": categories,
            "modules_with_data": sum(1 for m in modules_data if m["data_richness"] > 0),
            "modules_with_errors": sum(1 for m in modules_data if m["has_errors"]),
            "timestamp": datetime.now().isoformat(),
        }

        return sanitize_for_json(response_data)

    except Exception as e:
        logger.error(f"Critical error in list_modules endpoint: {e}", exc_info=True)
        # Return a minimal valid response instead of crashing
        return {
            "modules": [],
            "total_modules": 0,
            "enabled_modules": 0,
            "categories": {},
            "modules_with_data": 0,
            "modules_with_errors": 0,
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/api/reports/health")
async def health_report():
    """Structured health report including system, categories, and errors."""
    return {
        "system": {
            "status": state.system_status,
            "uptime": state.get_uptime(),
            "mt5_connected": state.mt5_connected,
            "model_loaded": state.model_loaded,
            "active_websockets": len(state.websocket_connections),
        },
        "metrics": state.system_metrics,
        "categories": state.get_category_summary(),
        "errors": state.errors[-20:],
        "warnings": state.warnings[-20:],
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/reports/state")
async def state_report():
    """Human-readable state report (plain text)."""
    cats = state.get_category_summary()
    lines = []
    lines.append("AI TRADING SYSTEM STATE REPORT")
    lines.append("".ljust(40, "="))
    lines.append(f"Status: {state.system_status}")
    lines.append(f"Uptime: {state.get_uptime()}")
    lines.append(f"MT5 Connected: {state.mt5_connected}")
    lines.append(f"Model Loaded: {state.model_loaded}")
    lines.append("")
    lines.append("MODULE CATEGORIES:")
    for cat, s in sorted(cats.items()):
        lines.append(f"  - {cat}: total={s['total']}, enabled={s['enabled']}, active={s['active']}, with_errors={s['with_errors']}")
    lines.append("")
    lines.append(f"Errors: {len(state.errors)} | Warnings: {len(state.warnings)} | Alerts: {len(state.alerts)}")
    lines.append(f"Active WS: {len(state.websocket_connections)}")
    return {"report": "\n".join(lines), "timestamp": datetime.now().isoformat()}

@app.get("/api/modules/{module_name}")
async def get_module_detailed_state(module_name: str):
    """Get detailed state for specific module"""
    if module_name not in state.module_states:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    module = state.module_states[module_name]
    return {
        "module": module_name,
        "state": module,
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/modules/{module_name}/toggle")
async def toggle_module(module_name: str):
    """Toggle module enabled/disabled state"""
    # Prefer orchestrator control when available
    orchestrator_ok = False
    try:
        from modules.core.module_system import ModuleOrchestrator  # type: ignore
        orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()  # type: ignore[attr-defined]
        current_state = bool(state.module_states.get(module_name, {}).get("enabled", True))
        if current_state:
            ok = bool(orch.disable_module(module_name, reason="User toggle"))
        else:
            ok = bool(orch.enable_module(module_name))
        orchestrator_ok = ok
    except Exception:
        orchestrator_ok = False

    if module_name not in state.module_states and not orchestrator_ok:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")

    # Sync from orchestrator if possible, else flip local state
    if not state._sync_modules_from_orchestrator():
        cur = state.module_states.get(module_name, {}).get("enabled", False)
        state.module_states.setdefault(module_name, {})["enabled"] = not cur

    action = "enabled" if state.module_states.get(module_name, {}).get("enabled", False) else "disabled"
    state.add_alert(f"Module {module_name} {action}", "info", module_name)

    await broadcast_system_state()
    return {"module": module_name, "enabled": state.module_states[module_name]["enabled"], "message": f"Module {action} successfully"}

@app.post("/api/modules/enable-all")
async def enable_all_modules():
    """Enable all known modules (frontend convenience)."""
    try:
        from modules.core.module_system import ModuleOrchestrator  # type: ignore
        orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()  # type: ignore[attr-defined]
        for name in list(getattr(orch, 'modules', {}).keys()):
            try:
                orch.enable_module(name)
            except Exception:
                pass
        state._sync_modules_from_orchestrator()
    except Exception:
        for name in list(state.module_states.keys()):
            state.module_states[name]["enabled"] = True
            state.module_states[name]["last_update"] = datetime.now().isoformat()
    await broadcast_system_state()
    return {
        "success": True,
        "enabled_modules": len(state.module_states),
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/modules/disable-all")
async def disable_all_modules():
    """Disable all known modules (frontend convenience)."""
    try:
        from modules.core.module_system import ModuleOrchestrator  # type: ignore
        orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()  # type: ignore[attr-defined]
        for name in list(getattr(orch, 'modules', {}).keys()):
            try:
                orch.disable_module(name, reason="User disable-all")
            except Exception:
                pass
        state._sync_modules_from_orchestrator()
    except Exception:
        for name in list(state.module_states.keys()):
            state.module_states[name]["enabled"] = False
            state.module_states[name]["last_update"] = datetime.now().isoformat()
    await broadcast_system_state()
    return {
        "success": True,
        "disabled_modules": len(state.module_states),
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/modules/refresh")
async def refresh_modules_from_registry():
    """Sync modules from orchestrator if available; otherwise reload registry."""
    try:
        if not state._sync_modules_from_orchestrator():
            state._merge_registry_modules('config/module_registry.yaml')
        await broadcast_system_state()
        return {
            "success": True,
            "total_modules": len(state.module_states),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to refresh modules: {e}")

@app.get("/api/performance")
async def get_performance_metrics():
    """Get detailed performance metrics"""
    return {
        "performance": state.performance_metrics,
        "risk_metrics": {
            "current_drawdown": state.performance_metrics.get("current_drawdown", 0.0),
            "max_drawdown": state.performance_metrics.get("max_drawdown", 0.0),
            "sharpe_ratio": state.performance_metrics.get("sharpe_ratio", 0.0),
            "win_rate": state.performance_metrics.get("win_rate", 0.0),
            "profit_factor": state.performance_metrics.get("profit_factor", 0.0),
        },
        "trading_stats": {
            "total_trades": state.performance_metrics.get("total_trades", 0),
            "winning_trades": state.performance_metrics.get("winning_trades", 0),
            "losing_trades": state.performance_metrics.get("losing_trades", 0),
            "trades_today": state.performance_metrics.get("trades_today", 0),
        },
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/alerts")
async def get_alerts(limit: int = Query(default=50, le=1000)):
    """Get system alerts"""
    return {
        "alerts": state.alerts[-limit:],
        "total_alerts": len(state.alerts),
        "timestamp": datetime.now().isoformat(),
    }

# ================== VISUALIZATION ENDPOINTS ==================
@app.get("/api/visualization/overview")
async def visualization_overview():
    """Return visualization summary (records, perf metrics, stats)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        viz = bus.get('visualization_data', 'BackendAPI', default={}) or {}
        # Trim performance_metrics to last 100 points
        pm = viz.get('performance_metrics', {}) if isinstance(viz, dict) else {}
        out = {
            'total_records': viz.get('total_records', 0),
            'statistics': viz.get('statistics', {}),
            'streaming_enabled': viz.get('streaming_enabled', False),
            'performance_metrics': pm,
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/visualization/dashboard")
async def visualization_dashboard():
    """Return dashboard data prepared by VisualizationInterface (if available)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        db = bus.get('dashboard_data', 'BackendAPI', default={}) or {}
        return {"success": True, "dashboard": db, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/visualization/alerts")
async def visualization_alerts(limit: int = Query(default=50, le=1000)):
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        al = bus.get('alert_timeline', 'BackendAPI', default=[]) or []
        return {"success": True, "alerts": al[-limit:], "count": len(al), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/visualization/trace")
async def visualization_trace(limit: int = Query(default=100, le=2000)):
    """Return recent decision trace/records if VisualizationInterface publishes them."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        viz = bus.get('visualization_data', 'BackendAPI', default={}) or {}
        trace = viz.get('decision_trace', []) if isinstance(viz, dict) else []
        if not isinstance(trace, list):
            trace = []
        return {"success": True, "trace": trace[-limit:], "count": len(trace), "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/visualization/reports")
async def visualization_reports():
    """Return analytics reports/performance report prepared by VisualizationInterface."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        analytics = bus.get('analytics_reports', 'BackendAPI', default={}) or {}
        return {"success": True, "analytics_reports": analytics, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/visualization/trade-charts")
async def visualization_trade_charts():
    """Return charts generated by TradeMapVisualizer (summaries + data)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "chart_statistics": bus.get('chart_statistics', 'BackendAPI', default={}) or {},
            "chart_history": bus.get('chart_history', 'BackendAPI', default=[]) or [],
            "trade_charts": bus.get('trade_charts', 'BackendAPI', default={}) or {},
            "performance_charts": bus.get('performance_charts', 'BackendAPI', default={}) or {},
            "dashboard_charts": bus.get('dashboard_charts', 'BackendAPI', default={}) or {},
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== FRONTEND COMPATIBILITY ENDPOINTS ==================
@app.get("/api/visualization-data")
async def get_visualization_data():
    """Consolidated visualization data for frontend analytics dashboard"""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()

        # Collect all visualization data
        viz_data = bus.get('visualization_data', 'BackendAPI', default={}) or {}

        # Build comprehensive response combining multiple data sources
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),

            # Core visualization data
            "total_records": viz_data.get('total_records', 0),
            "statistics": viz_data.get('statistics', {}),
            "streaming_enabled": viz_data.get('streaming_enabled', False),
            "performance_metrics": viz_data.get('performance_metrics', {}),
            "decision_trace": viz_data.get('decision_trace', []),

            # Regime analytics
            "regime_analytics": {
                "current_regime": bus.get('regime_prediction', 'BackendAPI', default=None),
                "regime_probabilities": bus.get('regime_probabilities', 'BackendAPI', default={}) or {},
                "regime_performance": bus.get('regime_performance', 'BackendAPI', default={}) or {},
                "regime_matrix_analysis": bus.get('regime_matrix_analysis', 'BackendAPI', default={}) or {},
            },

            # Market analytics
            "market_analytics": {
                "themes": bus.get('market_themes', 'BackendAPI', default={}) or {},
                "sentiment": bus.get('market_sentiment', 'BackendAPI', default={}) or {},
                "volatility": bus.get('volatility_analysis', 'BackendAPI', default={}) or {},
            },

            # Trading analytics
            "trading_analytics": {
                "recent_trades": bus.get('recent_trades', 'BackendAPI', default=[]) or [],
                "trade_performance": bus.get('trade_performance', 'BackendAPI', default={}) or {},
                "risk_metrics": bus.get('risk_metrics', 'BackendAPI', default={}) or {},
            },

            # Charts and visualizations
            "charts": {
                "trade_charts": bus.get('trade_charts', 'BackendAPI', default={}) or {},
                "performance_charts": bus.get('performance_charts', 'BackendAPI', default={}) or {},
                "dashboard_charts": bus.get('dashboard_charts', 'BackendAPI', default={}) or {},
                "chart_statistics": bus.get('chart_statistics', 'BackendAPI', default={}) or {},
            }
        }

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/api/dashboard-data")
async def get_dashboard_data():
    """Consolidated dashboard data for frontend analytics"""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()

        # Get dashboard data from InfoBus
        dashboard_data = bus.get('dashboard_data', 'BackendAPI', default={}) or {}

        # Enhanced dashboard response
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "dashboard": dashboard_data,

            # Additional dashboard metrics
            "system_health": {
                "uptime": state.get_uptime(),
                "session_id": state.current_session_id,
                "system_status": state.system_status,
                "mt5_connected": state.mt5_connected,
            },

            # Module status overview
            "modules_overview": {
                "total_modules": len(state.module_states),
                "active_modules": sum(1 for m in state.module_states.values() if m.get("enabled", False)),
                "module_health": {name: mod.get("status", "unknown") for name, mod in state.module_states.items()},
            },

            # Recent activity
            "recent_activity": {
                "recent_alerts": state.alerts[-10:] if state.alerts else [],
                "recent_trades": bus.get('recent_trades', 'BackendAPI', default=[]) or [],
                "latest_decisions": bus.get('latest_decisions', 'BackendAPI', default=[]) or [],
            }
        }

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/api/performance-metrics")
async def get_performance_metrics_v2():
    """Enhanced performance metrics endpoint for frontend compatibility"""
    try:
        # Get existing performance data
        perf_data = state.performance_metrics

        # Enhanced response with additional analytics
        response = {
            "success": True,
            "timestamp": datetime.now().isoformat(),

            # Core performance metrics (from existing endpoint)
            "performance": perf_data,

            # Risk metrics
            "risk_metrics": {
                "current_drawdown": perf_data.get("current_drawdown", 0.0),
                "max_drawdown": perf_data.get("max_drawdown", 0.0),
                "sharpe_ratio": perf_data.get("sharpe_ratio", 0.0),
                "win_rate": perf_data.get("win_rate", 0.0),
                "profit_factor": perf_data.get("profit_factor", 0.0),
                "var_95": perf_data.get("var_95", 0.0),
                "expected_shortfall": perf_data.get("expected_shortfall", 0.0),
            },

            # Trading statistics
            "trading_stats": {
                "total_trades": perf_data.get("total_trades", 0),
                "winning_trades": perf_data.get("winning_trades", 0),
                "losing_trades": perf_data.get("losing_trades", 0),
                "trades_today": perf_data.get("trades_today", 0),
                "avg_trade_duration": perf_data.get("avg_trade_duration", 0),
                "largest_win": perf_data.get("largest_win", 0.0),
                "largest_loss": perf_data.get("largest_loss", 0.0),
            },

            # Balance tracking
            "balance_metrics": {
                "current_balance": perf_data.get("current_balance", 0.0),
                "start_balance": perf_data.get("start_balance", 0.0),
                "peak_balance": perf_data.get("peak_balance", 0.0),
                "daily_pnl": perf_data.get("daily_pnl", 0.0),
                "total_pnl": perf_data.get("total_pnl", 0.0),
                "unrealized_pnl": perf_data.get("unrealized_pnl", 0.0),
            },

            # Performance trends (from InfoBus if available)
            "performance_trends": {},
        }

        # Add InfoBus performance data if available
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()

            response["performance_trends"] = {
                "performance_data": bus.get('performance_data', 'BackendAPI', default={}) or {},
                "performance_history": bus.get('performance_history', 'BackendAPI', default=[]) or [],
                "trade_analytics": bus.get('trade_analytics', 'BackendAPI', default={}) or {},
            }
        except Exception:
            pass  # InfoBus data is optional

        return response

    except Exception as e:
        return {"success": False, "error": str(e), "timestamp": datetime.now().isoformat()}

# ================== MT5 STATUS/ACCOUNT ==================
@app.get("/api/mt5/status")
async def mt5_status():
    try:
        info: Dict[str, Any] = {"connected": bool(state.mt5_connected)}
        if state.mt5_connected:
            try:
                ti = mt5.terminal_info()
                if ti is not None:
                    info.update({
                        "trade_allowed": bool(getattr(ti, 'trade_allowed', False)),
                        "community_connected": bool(getattr(ti, 'community_connected', False)),
                        "name": getattr(ti, 'name', None),
                        "company": getattr(ti, 'company', None),
                    })
            except Exception:
                pass
        return {"success": True, **info, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/mt5/account")
async def mt5_account():
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")
        ai = mt5.account_info()
        if ai is None:
            raise HTTPException(status_code=500, detail="Failed to retrieve MT5 account information")
        data = {
            "login": getattr(ai, 'login', None),
            "balance": getattr(ai, 'balance', None),
            "equity": getattr(ai, 'equity', None),
            "margin": getattr(ai, 'margin', None),
            "margin_free": getattr(ai, 'margin_free', None),
            "currency": getattr(ai, 'currency', None),
            "leverage": getattr(ai, 'leverage', None),
            "profit": getattr(ai, 'profit', None),
            "margin_level": getattr(ai, 'margin_level', None),
            "company": getattr(ai, 'company', None),
        }
        return {"success": True, "account": data, "timestamp": datetime.now().isoformat()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/mt5/chart-data/{symbol}")
async def get_mt5_chart_data(symbol: str, timeframe: str = "M5", count: int = 50):
    """Get real-time MT5 chart data for overview dashboard"""
    try:
        if not state.mt5_connected:
            # Return error if MT5 not connected (no simulated data)
            return {"success": False, "error": "MT5 not connected - no historical data available"}

        # Map timeframe string to MT5 constant
        timeframe_map = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1
        }

        tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)

        # Get rates from MT5
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

        if rates is None or len(rates) == 0:
            return {"success": False, "error": "MT5 not connected - no historical data available"}

        # Convert to frontend format
        chart_data = []
        for i, rate in enumerate(rates):
            # Normalize server timestamp to local time for display and include epoch for charting
            ts_utc = datetime.utcfromtimestamp(int(rate['time'])).replace(tzinfo=timezone.utc)
            ts_local = ts_utc.astimezone()
            chart_data.append({
                "time": ts_local.strftime("%H:%M"),  # human-readable local time
                "ts": int(ts_utc.timestamp()),        # epoch seconds
                "timestamp_ms": int(ts_utc.timestamp() * 1000),
                "open": float(rate['open']),
                "high": float(rate['high']),
                "low": float(rate['low']),
                "close": float(rate['close']),
                "volume": int(rate['tick_volume'])
            })

        return {
            "success": True,
            "symbol": symbol,
            "timeframe": timeframe,
            "data": chart_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        # Return error on exception
        return {"success": False, "error": f"Error fetching chart data: {str(e)}"}


@app.get("/api/mt5/positions")
async def get_mt5_positions():
    """Get current MT5 positions for overview dashboard"""
    try:
        if not state.mt5_connected:
            # Return error if MT5 not connected
            return {"success": False, "error": "MT5 not connected - no position data available"}

        # Get real positions from MT5
        positions = mt5.positions_get()
        if positions is None:
            positions = []

        position_data = []
        for pos in positions:
            position_data.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "profit": pos.profit,
                "sl": pos.sl,
                "tp": pos.tp,
                "has_sl": pos.sl > 0,
                "has_tp": pos.tp > 0,
                "time": datetime.fromtimestamp(pos.time).isoformat()
            })

        # Count positions without SL/TP
        missing_sl = sum(1 for p in position_data if not p["has_sl"])
        missing_tp = sum(1 for p in position_data if not p["has_tp"])

        return {
            "success": True,
            "positions": position_data,
            "total": len(position_data),
            "missing_sl": missing_sl,
            "missing_tp": missing_tp,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


@app.post("/api/mt5/positions/fix-sl-tp")
async def fix_positions_sl_tp():
    """Fix positions that are missing SL/TP - critical for network disconnect protection"""
    try:
        if not state.mt5_connected:
            return {"success": False, "error": "MT5 not connected"}

        # Load risk policy config
        import yaml
        try:
            with open('config/risk_policy.yaml', 'r') as f:
                risk_config = yaml.safe_load(f) or {}
                sl_tp_config = risk_config.get('sl_tp_settings', {})
        except Exception:
            sl_tp_config = {}

        if not sl_tp_config.get('fix_missing_sl_tp', True):
            return {"success": False, "error": "fix_missing_sl_tp is disabled in config"}

        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            return {"success": True, "message": "No positions to fix", "fixed": 0}

        fixed_count = 0
        results = []

        def pips_to_price(symbol: str, pips: float) -> float:
            """Convert pips to price distance"""
            sym_upper = symbol.upper()
            if 'XAU' in sym_upper or 'GOLD' in sym_upper:
                return pips * 0.01
            elif 'JPY' in sym_upper:
                return pips * 0.01
            else:
                return pips * 0.0001

        for pos in positions:
            current_sl = pos.sl
            current_tp = pos.tp
            
            needs_sl = current_sl <= 0 and sl_tp_config.get('auto_sl_enabled', True)
            needs_tp = current_tp <= 0 and sl_tp_config.get('auto_tp_enabled', True)
            
            if not needs_sl and not needs_tp:
                continue

            symbol = pos.symbol
            symbol_config = sl_tp_config.get(symbol, sl_tp_config.get('default', {}))
            sl_pips = symbol_config.get('stop_loss_pips', 50)
            tp_pips = symbol_config.get('take_profit_pips', 100)
            
            sl_distance = pips_to_price(symbol, sl_pips)
            tp_distance = pips_to_price(symbol, tp_pips)
            
            # Get symbol info for rounding
            symbol_info = mt5.symbol_info(symbol)
            digits = symbol_info.digits if symbol_info else 5
            
            new_sl = current_sl
            new_tp = current_tp
            is_buy = pos.type == mt5.ORDER_TYPE_BUY
            
            if is_buy:
                if needs_sl and sl_pips > 0:
                    new_sl = round(pos.price_open - sl_distance, digits)
                if needs_tp and tp_pips > 0:
                    new_tp = round(pos.price_open + tp_distance, digits)
            else:  # SELL
                if needs_sl and sl_pips > 0:
                    new_sl = round(pos.price_open + sl_distance, digits)
                if needs_tp and tp_pips > 0:
                    new_tp = round(pos.price_open - tp_distance, digits)

            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": pos.ticket,
                "sl": new_sl if new_sl > 0 else 0.0,
                "tp": new_tp if new_tp > 0 else 0.0,
            }
            
            result = mt5.order_send(request)
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                fixed_count += 1
                results.append({
                    "ticket": pos.ticket,
                    "symbol": symbol,
                    "sl": new_sl,
                    "tp": new_tp,
                    "ok": True
                })
                logger.info(f"[SL/TP] Fixed position {pos.ticket}: SL={new_sl:.5f} TP={new_tp:.5f}")
            else:
                error = result.retcode if result else "none"
                results.append({
                    "ticket": pos.ticket,
                    "symbol": symbol,
                    "ok": False,
                    "error": str(error)
                })
                logger.warning(f"[SL/TP] Failed to fix position {pos.ticket}: {error}")

        return {
            "success": True,
            "fixed": fixed_count,
            "total": len(positions),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"[SL/TP] Error fixing positions: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/mt5/deals/recent")
async def get_recent_mt5_deals(limit: int = 10):
    """Get recent MT5 deals for overview dashboard"""
    try:
        if not state.mt5_connected:
            # Return error if MT5 not connected
            return {"success": False, "error": "MT5 not connected - no deal history available"}

        # Get real deals from MT5
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now()
        deals = mt5.history_deals_get(from_date, to_date)

        if deals is None:
            deals = []

        # Sort by time and take most recent
        deals = sorted(deals, key=lambda x: x.time, reverse=True)[:limit]

        deal_data = []
        for deal in deals:
            deal_data.append({
                "ticket": deal.ticket,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "time": deal.time  # Unix timestamp - frontend will format it
            })

        return {
            "success": True,
            "deals": deal_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/mt5/symbols/active")
async def get_active_mt5_symbols():
    """Get active MT5 symbols with current prices"""
    try:
        if not state.mt5_connected:
            # Return error if MT5 not connected
            return {"success": False, "error": "MT5 not connected - no symbol data available"}

        # Get specific pairs requested by user
        major_pairs = ["EURUSD", "XAUUSD"]
        symbol_data = []

        for symbol in major_pairs:
            try:
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    symbol_data.append({
                        "symbol": symbol,
                        "bid": tick.bid,
                        "ask": tick.ask,
                        "spread": int((tick.ask - tick.bid) / mt5.symbol_info(symbol).point)
                    })
            except:
                continue

        return {
            "success": True,
            "symbols": symbol_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== AUDITING REPORT ENDPOINTS ==================
@app.get("/api/auditing/overview")
async def auditing_overview():
    """Return auditing coordinator overview (status, metrics, report preview)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        status = bus.get('audit_status', 'BackendAPI', default=None)
        metrics = bus.get('audit_metrics', 'BackendAPI', default={}) or {}
        report = bus.get('audit_report', 'BackendAPI', default=None)
        preview = report[:600] if isinstance(report, str) else None
        return {
            "success": True,
            "audit_status": status,
            "audit_metrics": metrics,
            "audit_report_preview": preview,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/auditing/trade-explanations")
async def auditing_trade_explanations():
    """Return trade explanations, metrics, and alerts from TradeExplanationAuditor."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        explanations = bus.get('trade_explanations', 'BackendAPI', default=[]) or []
        explanation_metrics = bus.get('explanation_metrics', 'BackendAPI', default={}) or {}
        audit_alerts = bus.get('audit_alerts', 'BackendAPI', default=[]) or []
        return {
            "success": True,
            "trade_explanations": explanations,
            "explanation_metrics": explanation_metrics,
            "audit_alerts": audit_alerts,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/auditing/thesis")
async def auditing_thesis():
    """Return thesis analysis/performance/alerts from TradeThesisTracker."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        thesis_analysis = bus.get('thesis_analysis', 'BackendAPI', default={}) or {}
        thesis_performance = bus.get('thesis_performance', 'BackendAPI', default={}) or {}
        thesis_alerts = bus.get('thesis_alerts', 'BackendAPI', default=[]) or []
        return {
            "success": True,
            "thesis_analysis": thesis_analysis,
            "thesis_performance": thesis_performance,
            "thesis_alerts": thesis_alerts,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== ANALYSIS ENDPOINTS ==================
@app.get("/api/analysis/overview")
async def analysis_overview():
    """Market analysis overview (regime, volatility/session, theme status)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "market_regime": bus.get('market_regime', 'BackendAPI', default=None),
            "volatility_level": bus.get('volatility_level', 'BackendAPI', default=None),
            "session_data": bus.get('session_data', 'BackendAPI', default={}) or {},
            "theme_detector_status": bus.get('theme_detector_status', 'BackendAPI', default=None),
            "theme_strength": bus.get('theme_strength', 'BackendAPI', default=None),
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/analysis/themes")
async def analysis_themes():
    """Detailed theme analytics from UnifiedMarket/ThemeDetector when present."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "active_theme": bus.get('market_theme', 'BackendAPI', default=None),
            "theme_strengths": bus.get('theme_strengths', 'BackendAPI', default={}) or {},
            "theme_transitions": bus.get('theme_transitions', 'BackendAPI', default=[]) or [],
            "unified_market_analysis": bus.get('unified_market_analysis', 'BackendAPI', default=None),
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/analysis/regime")
async def analysis_regime():
    """Regime details (type, probabilities, performance/matrix summaries)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "market_regime": bus.get('market_regime', 'BackendAPI', default=None),
            "regime_prediction": bus.get('regime_prediction', 'BackendAPI', default=None),
            "regime_probabilities": bus.get('regime_probabilities', 'BackendAPI', default=None) or bus.get('regime_probability', 'BackendAPI', default=None),
            "regime_performance": bus.get('regime_performance', 'BackendAPI', default=None),
            "regime_matrix_analysis": bus.get('regime_matrix_analysis', 'BackendAPI', default=None),
            "regime_matrix_status": bus.get('regime_matrix_status', 'BackendAPI', default=None),
            "regime_matrix_health": bus.get('regime_matrix_health', 'BackendAPI', default=None),
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== EXECUTOR ENDPOINTS ==================
@app.get("/api/executor/overview")
async def executor_overview():
    """Executor overview combining PositionManager and ExecutionQualityMonitor state."""
    try:
        pm = state.module_states.get('position_manager', {})
        eq = state.module_states.get('execution_monitor', {})
        # Positions can also be on the bus
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
            positions = bus.get('positions', 'BackendAPI', default=None)
        except Exception:
            positions = None
        out = {
            "position_manager": {
                "enabled": pm.get("enabled"),
                "status": pm.get("status"),
                "position_count": pm.get("position_count"),
                "total_exposure": pm.get("total_exposure"),
                "avg_holding_time": pm.get("avg_holding_time"),
                "instrument_exposures": pm.get("instrument_exposures", {}),
            },
            "execution_monitor": {
                "enabled": eq.get("enabled"),
                "status": eq.get("status"),
                "execution_quality": eq.get("execution_quality"),
                "slippage": eq.get("slippage"),
                "latency_ms": eq.get("latency_ms"),
                "fill_rate": eq.get("fill_rate"),
                "rejections": eq.get("rejections"),
                "partial_fills": eq.get("partial_fills"),
                "execution_costs": eq.get("execution_costs"),
            },
            "positions": positions if positions is not None else pm.get("open_positions", {}),
        }
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== LEGACY RISK ENDPOINT (for compatibility) ==================
@app.get("/api/risk/legacy")
async def risk_overview_legacy():
    """Legacy risk overview: drawdown/VAR/compliance/correlation from bus + state."""
    try:
        # Base from state
        perf = state.performance_metrics
        dd = {
            "current_drawdown": perf.get("current_drawdown", 0.0),
            "max_drawdown": perf.get("max_drawdown", 0.0),
            "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
            "win_rate": perf.get("win_rate", 0.0),
        }
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
            risk_metrics = bus.get('risk_metrics', 'BackendAPI', default={}) or {}
            compliance = bus.get('compliance', 'BackendAPI', default=None)
            correlation_risk = bus.get('correlation_risk', 'BackendAPI', default=None)
            correlation_matrix = bus.get('correlation_matrix', 'BackendAPI', default=None)
        except Exception:
            risk_metrics, compliance, correlation_risk, correlation_matrix = {}, None, None, None

        return {
            "success": True,
            "risk_metrics": {**risk_metrics, **dd},
            "compliance": compliance,
            "correlation": {
                "risk": correlation_risk,
                "matrix": correlation_matrix,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== STRATEGY/VOTING ENDPOINTS ==================
@app.get("/api/strategy/overview")
async def strategy_overview():
    """Strategy overview: trading signal + voting consensus."""
    try:
        try:
            from modules.utils.info_bus import InfoBusManager  # type: ignore
            bus = InfoBusManager.get_instance()
            trading_signal = bus.get('trading_signal', 'BackendAPI', default={}) or {}
            voting_consensus = bus.get('voting_consensus', 'BackendAPI', default=None)
            consensus_summary = bus.get('consensus_summary', 'BackendAPI', default=None)
            member_confidences = bus.get('member_confidences', 'BackendAPI', default=None)
        except Exception:
            trading_signal, voting_consensus, consensus_summary, member_confidences = {}, None, None, None

        return {
            "success": True,
            "trading_signal": trading_signal,
            "voting": {
                "consensus": voting_consensus,
                "summary": consensus_summary,
                "member_confidences": member_confidences,
            },
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== EXTERNAL MODULES ENDPOINTS ==================
@app.get("/api/external/market-data")
async def external_market_data():
    """Summarize latest market data provider outputs from the bus."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "environment_config": bus.get('environment_config', 'BackendAPI', default={}) or {},
            "market_data": bus.get('market_data', 'BackendAPI', default={}) or {},
            "multi_timeframe_data": bus.get('multi_timeframe_data', 'BackendAPI', default={}) or {},
            "step_idx": bus.get('step_idx', 'BackendAPI', default=None),
            "technical_indicators": bus.get('technical_indicators', 'BackendAPI', default=None),
            "volatility_level": bus.get('volatility_level', 'BackendAPI', default=None),
            "trading_session": bus.get('trading_session', 'BackendAPI', default=None),
        }
        # Reduce huge blobs: only include keys/meta sizes
        md = out.get("market_data", {})
        if isinstance(md, dict) and md:
            out["market_data_summary"] = {k: list(v.keys()) if isinstance(v, dict) else '...' for k, v in list(md.items())[:3]}
            out.pop("market_data", None)
        mtd = out.get("multi_timeframe_data", {})
        if isinstance(mtd, dict) and mtd:
            out["mtd_summary"] = {k: list(v.keys()) if isinstance(v, dict) else '...' for k, v in list(mtd.items())[:3]}
            out.pop("multi_timeframe_data", None)
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== FEATURES ENDPOINTS ==================
@app.get("/api/features/advanced")
async def features_advanced():
    """AdvancedFeatureEngine outputs (summaries to avoid huge blobs)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "feature_health": bus.get('feature_health', 'BackendAPI', default=None),
            "feature_thesis": bus.get('feature_thesis', 'BackendAPI', default=None),
            "feature_analysis": bus.get('feature_analysis', 'BackendAPI', default=None),
        }
        # Summaries for potentially large arrays
        def summarize(key: str):
            val = bus.get(key, 'BackendAPI', default=None)
            if isinstance(val, dict):
                return {k: (len(v) if hasattr(v, '__len__') else 'obj') for k, v in list(val.items())[:8]}
            return 'n/a' if val is None else 'available'
        out.update({
            "advanced_features": summarize('advanced_features'),
            "advanced_features_H1": summarize('advanced_features_H1'),
            "advanced_features_H4": summarize('advanced_features_H4'),
            "advanced_features_D1": summarize('advanced_features_D1'),
            "market_features": summarize('market_features'),
            "price_features": summarize('price_features'),
        })
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/features/multiscale")
async def features_multiscale():
    """MultiScaleFeatureEngine outputs (summaries for heavy tensors)."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        out = {
            "neural_health": bus.get('neural_health', 'BackendAPI', default=None),
            "neural_capabilities": bus.get('neural_capabilities', 'BackendAPI', default=None),
        }
        def summarize(key: str):
            val = bus.get(key, 'BackendAPI', default=None)
            if isinstance(val, dict):
                return {k: (len(v) if hasattr(v, '__len__') else 'obj') for k, v in list(val.items())[:8]}
            return 'n/a' if val is None else 'available'
        out.update({
            "multiscale_features": summarize('multiscale_features'),
            "attention_weights": summarize('attention_weights'),
            "feature_fusion": summarize('feature_fusion'),
            "neural_embeddings": summarize('neural_embeddings'),
        })
        return {"success": True, **out, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== MEMORY ENDPOINTS ==================
@app.get("/api/memory/overview")
async def memory_overview():
    """Get unified memory system overview - reads from training subprocess via persisted file"""
    try:
        # Use fallback to persisted file (training writes there)
        unified_metrics = get_bus_value_with_fallback('unified_metrics', 'BackendAPI', default={}) or {}
        memory_status = get_bus_value_with_fallback('unified_memory_status', 'BackendAPI', default={}) or {}

        # Extract core metrics
        overview = {
            "total_memories": unified_metrics.get("total_memories", 0),
            "memory_utilization": unified_metrics.get("memory_utilization", 0.0),
            "components_active": unified_metrics.get("components_active", 0),
            "processing_status": unified_metrics.get("processing_status", "unknown"),
            "health_status": unified_metrics.get("health_status", "unknown"),
            "components_enabled": memory_status.get("components_enabled", 0),
            "status": memory_status.get("status", "unknown")
        }

        # Sanitize to prevent inf/NaN JSON serialization errors
        overview = sanitize_for_json(overview)
        return {"success": True, **overview, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/memory/components")
async def memory_components():
    """Get all memory components status and data - reads from training subprocess via persisted file"""
    try:
        components = {
            "neural": {
                "neural_memory": get_bus_value_with_fallback('neural_memory', 'BackendAPI', default={}) or {},
                "attention_retrieval": get_bus_value_with_fallback('attention_retrieval', 'BackendAPI', default={}) or {},
                "memory_embedding": get_bus_value_with_fallback('memory_embedding', 'BackendAPI', default={}) or {},
                "importance_scoring": get_bus_value_with_fallback('importance_scoring', 'BackendAPI', default={}) or {}
            },
            "playbook": {
                "playbook_recall": get_bus_value_with_fallback('playbook_recall', 'BackendAPI', default={}) or {},
                "pattern_memory": get_bus_value_with_fallback('pattern_memory', 'BackendAPI', default={}) or {},
                "playbook_quality": get_bus_value_with_fallback('playbook_quality', 'BackendAPI', default={}) or {},
                "memory_analytics": get_bus_value_with_fallback('memory_analytics', 'BackendAPI', default={}) or {}
            },
            "mistakes": {
                "mistake_memory": get_bus_value_with_fallback('mistake_memory', 'BackendAPI', default={}) or {},
                "mistake_avoidance": get_bus_value_with_fallback('mistake_avoidance', 'BackendAPI', default={}) or {},
                "danger_zones": get_bus_value_with_fallback('danger_zones', 'BackendAPI', default={}) or {},
                "loss_prevention": get_bus_value_with_fallback('loss_prevention', 'BackendAPI', default={}) or {},
                "pattern_recognition": get_bus_value_with_fallback('pattern_recognition', 'BackendAPI', default={}) or {}
            },
            "replay": {
                "replay_sequences": get_bus_value_with_fallback('replay_sequences', 'BackendAPI', default={}) or {},
                "pattern_analysis": get_bus_value_with_fallback('pattern_analysis', 'BackendAPI', default={}) or {},
                "learning_progress": get_bus_value_with_fallback('learning_progress', 'BackendAPI', default={}) or {},
                "sequence_quality": get_bus_value_with_fallback('sequence_quality', 'BackendAPI', default={}) or {}
            },
            "compression": {
                "compressed_patterns": get_bus_value_with_fallback('compressed_patterns', 'BackendAPI', default={}) or {},
                "feature_importance": get_bus_value_with_fallback('feature_importance', 'BackendAPI', default={}) or {},
                "intuition_vector": get_bus_value_with_fallback('intuition_vector', 'BackendAPI', default={}) or {},
                "memory_compression": get_bus_value_with_fallback('memory_compression', 'BackendAPI', default={}) or {}
            },
            "budget": {
                "memory_allocation": get_bus_value_with_fallback('memory_allocation', 'BackendAPI', default={}) or {},
                "budget_optimization": get_bus_value_with_fallback('budget_optimization', 'BackendAPI', default={}) or {},
                "memory_efficiency": get_bus_value_with_fallback('memory_efficiency', 'BackendAPI', default={}) or {},
                "allocation_strategy": get_bus_value_with_fallback('allocation_strategy', 'BackendAPI', default={}) or {}
            }
        }

        # Sanitize to prevent inf/NaN JSON serialization errors
        components = sanitize_for_json(components)
        return {"success": True, "components": components, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/memory/patterns")
async def memory_patterns():
    """Get pattern analysis from memory components - reads from training subprocess via persisted file"""
    try:
        patterns = {
            "neural_patterns": {
                "attention_retrieval": get_bus_value_with_fallback('attention_retrieval', 'BackendAPI', default={}) or {},
                "memory_embedding": get_bus_value_with_fallback('memory_embedding', 'BackendAPI', default={}) or {}
            },
            "playbook_patterns": {
                "pattern_memory": get_bus_value_with_fallback('pattern_memory', 'BackendAPI', default={}) or {},
                "pattern_analysis": get_bus_value_with_fallback('pattern_analysis', 'BackendAPI', default={}) or {}
            },
            "compressed_patterns": get_bus_value_with_fallback('compressed_patterns', 'BackendAPI', default={}) or {},
            "pattern_recognition": get_bus_value_with_fallback('pattern_recognition', 'BackendAPI', default={}) or {}
        }

        # Sanitize to prevent inf/NaN JSON serialization errors
        patterns = sanitize_for_json(patterns)
        return {"success": True, "patterns": patterns, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/memory/mistakes")
async def memory_mistakes():
    """Get mistake analysis and danger zones - reads from training subprocess via persisted file"""
    try:
        mistakes = {
            "mistake_memory": get_bus_value_with_fallback('mistake_memory', 'BackendAPI', default={}) or {},
            "mistake_avoidance": get_bus_value_with_fallback('mistake_avoidance', 'BackendAPI', default={}) or {},
            "danger_zones": get_bus_value_with_fallback('danger_zones', 'BackendAPI', default={}) or {},
            "loss_prevention": get_bus_value_with_fallback('loss_prevention', 'BackendAPI', default={}) or {},
            "pattern_recognition": get_bus_value_with_fallback('pattern_recognition', 'BackendAPI', default={}) or {}
        }

        # Sanitize to prevent inf/NaN JSON serialization errors
        mistakes = sanitize_for_json(mistakes)
        return {"success": True, "mistakes": mistakes, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/memory/performance")
async def memory_performance():
    """Get memory performance metrics and health data - reads from training subprocess via persisted file"""
    try:
        performance = {
            "overview": get_bus_value_with_fallback('unified_metrics', 'BackendAPI', default={}) or {},
            "neural_performance": {
                "neural_memory": get_bus_value_with_fallback('neural_memory', 'BackendAPI', default={}) or {},
                "importance_scoring": get_bus_value_with_fallback('importance_scoring', 'BackendAPI', default={}) or {}
            },
            "playbook_performance": {
                "playbook_quality": get_bus_value_with_fallback('playbook_quality', 'BackendAPI', default={}) or {},
                "memory_analytics": get_bus_value_with_fallback('memory_analytics', 'BackendAPI', default={}) or {}
            },
            "compression_performance": {
                "memory_compression": get_bus_value_with_fallback('memory_compression', 'BackendAPI', default={}) or {},
                "feature_importance": get_bus_value_with_fallback('feature_importance', 'BackendAPI', default={}) or {}
            },
            "budget_performance": {
                "budget_optimization": get_bus_value_with_fallback('budget_optimization', 'BackendAPI', default={}) or {},
                "memory_efficiency": get_bus_value_with_fallback('memory_efficiency', 'BackendAPI', default={}) or {}
            }
        }

        # Sanitize to prevent inf/NaN JSON serialization errors
        performance = sanitize_for_json(performance)
        return {"success": True, "performance": performance, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== RISK ENDPOINTS ==================
@app.get("/api/risk/overview")
async def risk_overview_enhanced():
    """Enhanced risk overview with comprehensive risk metrics"""
    try:
        # Base performance metrics
        perf = state.performance_metrics
        base_metrics = {
            "current_drawdown": perf.get("current_drawdown", 0.0),
            "max_drawdown": perf.get("max_drawdown", 0.0),
            "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
            "win_rate": perf.get("win_rate", 0.0),
        }

        # Risk system metrics from InfoBus (with fallback to persisted file)
        risk_metrics = get_bus_value_with_fallback('risk_metrics', 'BackendAPI', default={}) or {}
        risk_level = get_bus_value_with_fallback('risk_level', 'BackendAPI', default='UNKNOWN')
        risk_scale = get_bus_value_with_fallback('risk_scale', 'BackendAPI', default=1.0)
        risk_assessment = get_bus_value_with_fallback('risk_assessment', 'BackendAPI', default={}) or {}

        # Module states for additional context
        risk_controller = state.module_states.get("risk_controller", {})
        drawdown_rescue = state.module_states.get("drawdown_rescue", {})

        overview = {
            **base_metrics,
            **risk_metrics,
            **risk_assessment,
            "risk_level": risk_level,
            "risk_scale": risk_scale,
            "system_status": risk_controller.get("status", "unknown"),
            "rescue_active": drawdown_rescue.get("rescue_active", False),
            "var_95": risk_controller.get("var_95", 0.0),
            "var_99": risk_controller.get("var_99", 0.0),
            "volatility_ratio": risk_controller.get("volatility_ratio", 1.0),
            "risk_budget_used": risk_controller.get("risk_budget_used", 0.0)
        }

        return {"success": True, **overview, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/anomalies")
async def risk_anomalies():
    """Get anomaly detection data and alerts - reads from training subprocess via persisted file"""
    try:
        anomalies = {
            "anomaly_detection": get_bus_value_with_fallback('anomaly_detection', 'BackendAPI', default={}) or {},
            "anomaly_alerts": get_bus_value_with_fallback('anomaly_alerts', 'BackendAPI', default=[]) or [],
            "anomaly_score": get_bus_value_with_fallback('anomaly_score', 'BackendAPI', default=0.0),
            "anomaly_threshold": get_bus_value_with_fallback('anomaly_threshold', 'BackendAPI', default=0.8),
            "detection_mode": get_bus_value_with_fallback('detection_mode', 'BackendAPI', default='NORMAL'),
            "anomaly_history": get_bus_value_with_fallback('anomaly_history', 'BackendAPI', default=[]) or [],
            "system_health": get_bus_value_with_fallback('system_health', 'BackendAPI', default={}) or {}
        }

        return {"success": True, "anomalies": anomalies, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/compliance")
async def risk_compliance():
    """Get compliance monitoring data and violations - reads from training subprocess via persisted file"""
    try:
        compliance = {
            "compliance_status": get_bus_value_with_fallback('compliance', 'BackendAPI', default={}) or {},
            "trade_compliance": get_bus_value_with_fallback('trade_compliance', 'BackendAPI', default={}) or {},
            "compliance_violations": get_bus_value_with_fallback('compliance_violations', 'BackendAPI', default=[]) or [],
            "risk_limits": get_bus_value_with_fallback('risk_limits', 'BackendAPI', default={}) or {},
            "position_compliance": get_bus_value_with_fallback('position_compliance', 'BackendAPI', default={}) or {},
            "leverage_compliance": get_bus_value_with_fallback('leverage_compliance', 'BackendAPI', default={}) or {},
            "daily_limits": get_bus_value_with_fallback('daily_limits', 'BackendAPI', default={}) or {}
        }

        return {"success": True, "compliance": compliance, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/drawdown")
async def risk_drawdown():
    """Get drawdown monitoring and rescue system data - reads from training subprocess via persisted file"""
    try:
        # Module state for drawdown rescue
        drawdown_state = state.module_states.get("drawdown_rescue", {})

        drawdown = {
            "drawdown_status": get_bus_value_with_fallback('drawdown_status', 'BackendAPI', default={}) or {},
            "rescue_status": get_bus_value_with_fallback('rescue_status', 'BackendAPI', default={}) or {},
            "drawdown_analysis": get_bus_value_with_fallback('drawdown_analysis', 'BackendAPI', default={}) or {},
            "recovery_progress": get_bus_value_with_fallback('recovery_progress', 'BackendAPI', default={}) or {},
            "drawdown_history": get_bus_value_with_fallback('drawdown_history', 'BackendAPI', default=[]) or [],
            "rescue_triggers": get_bus_value_with_fallback('rescue_triggers', 'BackendAPI', default=[]) or [],
            "velocity_analysis": get_bus_value_with_fallback('velocity_analysis', 'BackendAPI', default={}) or {},
            "rescue_active": drawdown_state.get("rescue_active", False),
            "current_drawdown": drawdown_state.get("current_drawdown", 0.0),
            "max_drawdown": drawdown_state.get("max_drawdown", 0.0)
        }

        return {"success": True, "drawdown": drawdown, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/execution")
async def risk_execution():
    """Get execution quality monitoring data - reads from training subprocess via persisted file"""
    try:
        execution = {
            "execution_quality": get_bus_value_with_fallback('execution_quality', 'BackendAPI', default={}) or {},
            "execution_metrics": get_bus_value_with_fallback('execution_metrics', 'BackendAPI', default={}) or {},
            "execution_alerts": get_bus_value_with_fallback('execution_alerts', 'BackendAPI', default=[]) or [],
            "slippage_analysis": get_bus_value_with_fallback('slippage_analysis', 'BackendAPI', default={}) or {},
            "latency_metrics": get_bus_value_with_fallback('latency_metrics', 'BackendAPI', default={}) or {},
            "fill_rate_analysis": get_bus_value_with_fallback('fill_rate_analysis', 'BackendAPI', default={}) or {},
            "execution_vote": get_bus_value_with_fallback('execution_vote', 'BackendAPI', default='ABSTAIN'),
            "quality_score": get_bus_value_with_fallback('quality_score', 'BackendAPI', default=0.0)
        }

        return {"success": True, "execution": execution, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/portfolio")
async def risk_portfolio():
    """Get portfolio risk system data and correlation analysis - reads from training subprocess via persisted file"""
    try:
        portfolio = {
            "portfolio_risk": get_bus_value_with_fallback('portfolio_risk', 'BackendAPI', default={}) or {},
            "correlation_matrix": get_bus_value_with_fallback('correlation_matrix', 'BackendAPI', default={}) or {},
            "correlation_risk": get_bus_value_with_fallback('correlation_risk', 'BackendAPI', default={}) or {},
            "position_risk": get_bus_value_with_fallback('position_risk', 'BackendAPI', default={}) or {},
            "var_analysis": get_bus_value_with_fallback('var_analysis', 'BackendAPI', default={}) or {},
            "risk_attribution": get_bus_value_with_fallback('risk_attribution', 'BackendAPI', default={}) or {},
            "exposure_analysis": get_bus_value_with_fallback('exposure_analysis', 'BackendAPI', default={}) or {},
            "diversification_metrics": get_bus_value_with_fallback('diversification_metrics', 'BackendAPI', default={}) or {}
        }

        return {"success": True, "portfolio": portfolio, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/dynamic")
async def risk_dynamic():
    """Get dynamic risk controller data and scaling metrics - reads from training subprocess via persisted file"""
    try:
        # Module state for additional context
        risk_controller = state.module_states.get("risk_controller", {})

        dynamic = {
            "dynamic_risk": get_bus_value_with_fallback('dynamic_risk', 'BackendAPI', default={}) or {},
            "risk_scaling": get_bus_value_with_fallback('risk_scaling', 'BackendAPI', default={}) or {},
            "volatility_analysis": get_bus_value_with_fallback('volatility_analysis', 'BackendAPI', default={}) or {},
            "risk_adjustments": get_bus_value_with_fallback('risk_adjustments', 'BackendAPI', default=[]) or [],
            "control_mode": get_bus_value_with_fallback('control_mode', 'BackendAPI', default='NORMAL'),
            "scaling_history": get_bus_value_with_fallback('scaling_history', 'BackendAPI', default=[]) or [],
            "risk_level": risk_controller.get("risk_level", "NORMAL"),
            "risk_scale": risk_controller.get("risk_scale", 1.0),
            "volatility": risk_controller.get("volatility", {}),
            "freeze_counter": risk_controller.get("freeze_counter", 0)
        }

        return {"success": True, "dynamic": dynamic, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/risk/alerts")
async def risk_alerts():
    """Get all risk-related alerts and notifications - reads from training subprocess via persisted file"""
    try:
        # Collect alerts from all risk modules (with fallback to persisted file)
        alerts = {
            "anomaly_alerts": get_bus_value_with_fallback('anomaly_alerts', 'BackendAPI', default=[]) or [],
            "compliance_alerts": get_bus_value_with_fallback('compliance_violations', 'BackendAPI', default=[]) or [],
            "drawdown_alerts": get_bus_value_with_fallback('rescue_triggers', 'BackendAPI', default=[]) or [],
            "execution_alerts": get_bus_value_with_fallback('execution_alerts', 'BackendAPI', default=[]) or [],
            "portfolio_alerts": get_bus_value_with_fallback('portfolio_alerts', 'BackendAPI', default=[]) or [],
            "risk_alerts": get_bus_value_with_fallback('risk_alerts', 'BackendAPI', default=[]) or [],
            "system_alerts": [alert for alert in state.alerts if alert.get('category') == 'risk']
        }

        # Calculate alert summary
        total_alerts = sum(len(alert_list) for alert_list in alerts.values())
        critical_count = sum(1 for alert_list in alerts.values()
                           for alert in alert_list if alert.get('severity') == 'critical')

        summary = {
            "total_alerts": total_alerts,
            "critical_count": critical_count,
            "last_update": datetime.now().isoformat()
        }

        return {"success": True, "alerts": alerts, "summary": summary, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════
# VOTING SYSTEM API ENDPOINTS v1.0
# ═══════════════════════════════════════════════════════════════════

@app.get("/api/voting/overview")
async def voting_overview():
    """Get comprehensive voting system overview - reads from training subprocess via persisted file"""
    try:
        # Core voting metrics (with fallback to persisted file)
        voting_metrics = get_bus_value_with_fallback('voting_metrics', 'BackendAPI', default={}) or {}
        decision_coordination = get_bus_value_with_fallback('decision_coordination', 'BackendAPI', default={}) or {}
        consensus_score = get_bus_value_with_fallback('consensus_score', 'BackendAPI', default=0.0) or 0.0
        consensus_components = get_bus_value_with_fallback('consensus_components', 'BackendAPI', default={}) or {}
        pipeline_stats = get_bus_value_with_fallback('pipeline_stats', 'BackendAPI', default={}) or {}
        consensus_decision = get_bus_value_with_fallback('consensus_decision', 'BackendAPI', default={}) or {}
        voting_result = get_bus_value_with_fallback('voting_result', 'BackendAPI', default={}) or {}

        # Calculate health status based on metrics
        successful_ticks = voting_metrics.get('successful_ticks', 0)
        total_ticks = voting_metrics.get('total_ticks', 1)
        success_rate = successful_ticks / max(total_ticks, 1)

        if success_rate >= 0.9:
            health_status = "healthy"
        elif success_rate >= 0.7:
            health_status = "warning"
        else:
            health_status = "critical"

        # Active components count
        components_active = 0
        if get_bus_value_with_fallback('committee_members', 'BackendAPI'):
            components_active += 1
        if get_bus_value_with_fallback('consensus_score', 'BackendAPI') is not None:
            components_active += 1
        if get_bus_value_with_fallback('collusion_score', 'BackendAPI') is not None:
            components_active += 1
        if get_bus_value_with_fallback('aligned_weights', 'BackendAPI'):
            components_active += 1
        if get_bus_value_with_fallback('sampling_uncertainty', 'BackendAPI') is not None:
            components_active += 1
        if get_bus_value_with_fallback('trade_vote_v2', 'BackendAPI'):
            components_active += 1

        overview = {
            "total_decisions": total_ticks,
            "successful_decisions": successful_ticks,
            "success_rate": success_rate,
            "components_active": components_active,
            "health_status": health_status,
            "current_consensus": consensus_score,
            "processing_time_ms": voting_metrics.get('avg_processing_time_ms', 0.0),
            "decision_id": decision_coordination.get('decision_id', 'none'),
            "consensus_decision": consensus_decision,
            "voting_result": voting_result,
            "last_update": datetime.now().isoformat()
        }

        return {"success": True, **overview, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/committee")
async def voting_committee():
    """Get voting committee data and member analytics - reads from training subprocess via persisted file"""
    try:
        # Committee data (with fallback to persisted file)
        committee_data = {
            "members": get_bus_value_with_fallback('committee_members', 'BackendAPI', default=[]) or [],
            "proposal_vectors": get_bus_value_with_fallback('proposal_vectors', 'BackendAPI', default=[]) or [],
            "member_confidences": get_bus_value_with_fallback('member_confidences_ordered', 'BackendAPI', default=[]) or [],
            "committee_consensus": get_bus_value_with_fallback('committee_consensus', 'BackendAPI', default={}) or {},
            "committee_votes": get_bus_value_with_fallback('committee_votes', 'BackendAPI', default=[]) or [],
        }

        # Member analytics from InfoBus
        member_analytics = get_bus_value_with_fallback('member_analytics', 'BackendAPI', default=[]) or []

        committee_summary = {
            "total_members": len(committee_data["members"]),
            "active_members": len([m for m in committee_data["members"] if m.get("active", True)]),
            "avg_confidence": sum(c for c in committee_data["member_confidences"]) / max(len(committee_data["member_confidences"]), 1),
            "consensus_strength": committee_data["committee_consensus"].get("strength", 0.0),
            "last_vote_time": get_bus_value_with_fallback('last_vote_time', 'BackendAPI', default=datetime.now().isoformat())
        }

        return {
            "success": True,
            "committee": {
                "data": committee_data,
                "analytics": member_analytics,
                "summary": committee_summary
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/consensus")
async def voting_consensus():
    """Get consensus detection data and analysis"""
    try:
        from modules.utils.info_bus import InfoBusManager
        bus = InfoBusManager.get_instance()

        # Consensus data
        consensus_score = bus.get('consensus_score', 'VotingKernel', default=0.0)
        consensus_components = bus.get('consensus_components', 'VotingKernel', default={}) or {}
        voting_consensus = bus.get('voting_consensus', 'VotingKernel', default={}) or {}

        # Component breakdown
        consensus_breakdown = {
            "directional": consensus_components.get('directional_consensus', 0.0),
            "magnitude": consensus_components.get('magnitude_consensus', 0.0),
            "confidence": consensus_components.get('confidence_consensus', 0.0),
            "temporal": consensus_components.get('temporal_stability', 0.0),
            "network": consensus_components.get('network_consensus', 0.0)
        }

        # Consensus analytics
        consensus_analytics = {
            "overall_score": consensus_score or 0.0,
            "quality": voting_consensus.get('quality', 0.0),
            "stability": consensus_components.get('temporal_stability', 0.0),
            "agreement_level": "high" if consensus_score and consensus_score > 0.7 else "medium" if consensus_score and consensus_score > 0.4 else "low",
            "trend": "improving" if consensus_score and consensus_score > 0.6 else "stable",
            "reliability": voting_consensus.get('reliability', 0.0)
        }

        return {
            "success": True,
            "consensus": {
                "score": consensus_score,
                "components": consensus_components,
                "breakdown": consensus_breakdown,
                "analytics": consensus_analytics,
                "raw_data": voting_consensus
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/collusion")
async def voting_collusion():
    """Get collusion detection and anti-manipulation data - reads from training subprocess via persisted file"""
    try:
        # Collusion data (with fallback to persisted file)
        collusion_score = get_bus_value_with_fallback('collusion_score', 'BackendAPI', default=0.0)
        suspicious_pairs = get_bus_value_with_fallback('suspicious_pairs', 'BackendAPI', default=[]) or []

        # Collusion analysis
        collusion_analysis = {
            "risk_level": "high" if collusion_score and collusion_score > 0.8 else "medium" if collusion_score and collusion_score > 0.5 else "low",
            "suspicious_pairs_count": len(suspicious_pairs),
            "detection_status": "active",
            "last_scan": datetime.now().isoformat(),
            "threat_score": collusion_score or 0.0
        }

        # Member integrity scores from InfoBus (with fallback)
        member_integrity = get_bus_value_with_fallback('member_integrity', 'BackendAPI', default=[]) or []

        # Collusion alerts from InfoBus (with fallback)
        collusion_alerts = get_bus_value_with_fallback('collusion_alerts', 'BackendAPI', default=[]) or []

        return {
            "success": True,
            "collusion": {
                "score": collusion_score,
                "suspicious_pairs": suspicious_pairs,
                "analysis": collusion_analysis,
                "member_integrity": member_integrity,
                "alerts": collusion_alerts
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/alignment")
async def voting_alignment():
    """Get time horizon alignment and weight distribution data - reads from training subprocess via persisted file"""
    try:
        # Weight alignment data (with fallback to persisted file)
        voting_weights = get_bus_value_with_fallback('voting_weights', 'BackendAPI', default=[]) or []
        aligned_weights = get_bus_value_with_fallback('aligned_weights', 'BackendAPI', default=[]) or []

        # Time horizon analysis from InfoBus (with fallback)
        alignment_analysis = get_bus_value_with_fallback('alignment_analysis', 'BackendAPI', default={
            "raw_weights": voting_weights,
            "aligned_weights": aligned_weights,
            "alignment_quality": 0.0,
            "temporal_coherence": 0.0,
            "horizon_distribution": {}
        }) or {
            "raw_weights": voting_weights,
            "aligned_weights": aligned_weights,
            "alignment_quality": 0.0,
            "temporal_coherence": 0.0,
            "horizon_distribution": {}
        }

        # Alignment metrics from InfoBus (with fallback)
        alignment_metrics = get_bus_value_with_fallback('alignment_metrics', 'BackendAPI', default={
            "total_weights": len(voting_weights),
            "alignment_strength": 0.0,
            "temporal_stability": 0.0,
            "weight_variance": 0.0,
            "optimization_score": 0.0
        }) or {
            "total_weights": len(voting_weights),
            "alignment_strength": 0.0,
            "temporal_stability": 0.0,
            "weight_variance": 0.0,
            "optimization_score": 0.0
        }

        # Horizon breakdown from InfoBus (with fallback)
        horizon_breakdown = get_bus_value_with_fallback('horizon_breakdown', 'BackendAPI', default=[]) or []

        return {
            "success": True,
            "alignment": {
                "analysis": alignment_analysis,
                "metrics": alignment_metrics,
                "horizon_breakdown": horizon_breakdown
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/sampling")
async def voting_sampling():
    """Get alternative reality sampling and uncertainty data - reads from training subprocess via persisted file"""
    try:
        # Sampling data (with fallback to persisted file)
        sampling_uncertainty = get_bus_value_with_fallback('sampling_uncertainty', 'BackendAPI', default=0.0)
        fragility = get_bus_value_with_fallback('fragility', 'BackendAPI', default=0.0)
        effective_samples = get_bus_value_with_fallback('effective_samples', 'BackendAPI', default=0)

        # Uncertainty analysis from InfoBus (with fallback)
        uncertainty_analysis = get_bus_value_with_fallback('uncertainty_analysis', 'BackendAPI', default={
            "uncertainty_level": sampling_uncertainty or 0.0,
            "fragility_score": fragility or 0.0,
            "robustness": 1.0 - (fragility or 0.0) if fragility else 0.0,
            "confidence_interval": 0.0,
            "sample_diversity": 0.0,
            "stability_measure": 0.0
        }) or {
            "uncertainty_level": sampling_uncertainty or 0.0,
            "fragility_score": fragility or 0.0,
            "robustness": 1.0 - (fragility or 0.0) if fragility else 0.0,
            "confidence_interval": 0.0,
            "sample_diversity": 0.0,
            "stability_measure": 0.0
        }

        # Sampling metrics from InfoBus (with fallback)
        sampling_metrics = get_bus_value_with_fallback('sampling_metrics', 'BackendAPI', default={
            "total_samples": effective_samples or 0,
            "effective_samples": effective_samples or 0,
            "sample_quality": 0.0,
            "convergence_rate": 0.0,
            "exploration_breadth": 0.0
        }) or {
            "total_samples": effective_samples or 0,
            "effective_samples": effective_samples or 0,
            "sample_quality": 0.0,
            "convergence_rate": 0.0,
            "exploration_breadth": 0.0
        }

        # Risk assessment
        uncertainty_level = sampling_uncertainty or 0.0
        risk_assessment = {
            "risk_level": "high" if uncertainty_level > 0.7 else "medium" if uncertainty_level > 0.4 else "low",
            "recommendation": "caution" if uncertainty_level > 0.6 else "proceed" if uncertainty_level < 0.3 else "monitor",
            "confidence_score": 1.0 - uncertainty_level,
            "decision_quality": "high" if uncertainty_level < 0.3 else "medium" if uncertainty_level < 0.6 else "low"
        }

        return {
            "success": True,
            "sampling": {
                "analysis": uncertainty_analysis,
                "metrics": sampling_metrics,
                "risk_assessment": risk_assessment
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/strategy")
async def voting_strategy():
    """Get strategy arbiter and final gating data - reads from training subprocess via persisted file"""
    try:
        # Strategy data (with fallback to persisted file)
        trade_vote_v2 = get_bus_value_with_fallback('trade_vote_v2', 'BackendAPI', default={}) or {}
        signals = get_bus_value_with_fallback('signals', 'BackendAPI', default={}) or {}

        decision_value = str(trade_vote_v2.get('decision', 'none') or 'none').lower()
        if decision_value in ('pass', 'passed', 'approve'):
            gating_status_default = 'passed'
        elif decision_value in ('block', 'blocked', 'reject'):
            gating_status_default = 'blocked'
        else:
            gating_status_default = decision_value

        # Strategy analysis from InfoBus (with fallback)
        strategy_analysis = get_bus_value_with_fallback('strategy_analysis', 'BackendAPI', default={
            "final_decision": trade_vote_v2.get('decision', 'none'),
            "confidence": trade_vote_v2.get('confidence', 0.0),
            "signal_strength": signals.get('strength', 0.0),
            "gating_status": gating_status_default,
            "arbitration_quality": 0.0
        }) or {
            "final_decision": trade_vote_v2.get('decision', 'none'),
            "confidence": trade_vote_v2.get('confidence', 0.0),
            "signal_strength": signals.get('strength', 0.0),
            "gating_status": gating_status_default,
            "arbitration_quality": 0.0
        }

        # Signal breakdown from InfoBus (with fallback)
        signal_breakdown = get_bus_value_with_fallback('signal_breakdown', 'BackendAPI', default={
            "primary_signal": signals.get('primary', 'neutral'),
            "secondary_signals": signals.get('secondary', []),
            "signal_coherence": 0.0,
            "cross_validation": 0.0,
            "execution_readiness": 0.0
        }) or {
            "primary_signal": signals.get('primary', 'neutral'),
            "secondary_signals": signals.get('secondary', []),
            "signal_coherence": 0.0,
            "cross_validation": 0.0,
            "execution_readiness": 0.0
        }

        # Strategy metrics from InfoBus (with fallback)
        strategy_metrics = get_bus_value_with_fallback('strategy_metrics', 'BackendAPI', default={
            "arbitration_success_rate": 0.0,
            "signal_accuracy": 0.0,
            "gating_efficiency": 0.0,
            "decision_latency_ms": trade_vote_v2.get('processing_time', 0),
            "quality_score": 0.0
        }) or {
            "arbitration_success_rate": 0.0,
            "signal_accuracy": 0.0,
            "gating_efficiency": 0.0,
            "decision_latency_ms": trade_vote_v2.get('processing_time', 0),
            "quality_score": 0.0
        }

        # Performance tracking from InfoBus (with fallback)
        performance_tracking = get_bus_value_with_fallback('performance_tracking', 'BackendAPI', default={
            "total_arbitrations": 0,
            "successful_arbitrations": 0,
            "blocked_decisions": 0,
            "avg_confidence": 0.0,
            "last_arbitration": None
        }) or {
            "total_arbitrations": 0,
            "successful_arbitrations": 0,
            "blocked_decisions": 0,
            "avg_confidence": 0.0,
            "last_arbitration": None
        }

        return {
            "success": True,
            "strategy": {
                "trade_vote": trade_vote_v2,
                "signals": signals,
                "analysis": strategy_analysis,
                "breakdown": signal_breakdown,
                "metrics": strategy_metrics,
                "performance": performance_tracking
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/timeline")
async def voting_timeline():
    """Get voting pipeline timeline and performance data - reads from training subprocess via persisted file"""
    try:
        # Timeline data (with fallback to persisted file)
        kernel_timeline = get_bus_value_with_fallback('voting/kernel_timeline', 'BackendAPI', default={}) or {}
        pipeline_stats = get_bus_value_with_fallback('pipeline_stats', 'BackendAPI', default={}) or {}

        # Extract timeline
        timeline = kernel_timeline.get('timeline', [])
        decision_id = kernel_timeline.get('decision_id', 'none')

        # Performance analysis
        performance_analysis = {
            "total_stages": len(timeline),
            "successful_stages": len([s for s in timeline if s.get('status') == 'success']),
            "failed_stages": len([s for s in timeline if s.get('status') == 'error']),
            "avg_stage_time": sum(s.get('duration_ms', 0) for s in timeline) / max(len(timeline), 1),
            "bottleneck_stage": max(timeline, key=lambda x: x.get('duration_ms', 0)).get('stage', 'none') if timeline else 'none'
        }

        # Stage breakdown
        stage_breakdown = []
        stage_names = ['committee', 'consensus', 'collusion', 'horizon', 'sampling', 'arbiter']
        for stage in stage_names:
            stage_data = next((s for s in timeline if s.get('stage') == stage), {})
            breakdown = {
                "stage": stage,
                "status": stage_data.get('status', 'unknown'),
                "duration_ms": stage_data.get('duration_ms', 0),
                "success_rate": pipeline_stats.get('module_success_rates', {}).get(stage, {}).get('success', 0) /
                               max(pipeline_stats.get('module_success_rates', {}).get(stage, {}).get('total', 1), 1)
            }
            stage_breakdown.append(breakdown)

        return {
            "success": True,
            "timeline": {
                "decision_id": decision_id,
                "stages": timeline,
                "analysis": performance_analysis,
                "breakdown": stage_breakdown,
                "pipeline_stats": pipeline_stats
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/external/news-sentiment")
async def external_news_sentiment():
    """Return news sentiment snapshot from bus."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        ns = bus.get('news_sentiment', 'BackendAPI', default={}) or {}
        trend = bus.get('sentiment_trend', 'BackendAPI', default=None)
        alerts = bus.get('sentiment_alerts', 'BackendAPI', default=None)
        return {"success": True, "news_sentiment": ns, "trend": trend, "alerts": alerts, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/external/session")
async def external_session():
    """Return session labels/context from SessionManager via bus."""
    try:
        from modules.utils.info_bus import InfoBusManager  # type: ignore
        bus = InfoBusManager.get_instance()
        data = {
            "trading_session": bus.get('trading_session', 'BackendAPI', default=None),
            "session_type": bus.get('session_type', 'BackendAPI', default=None),
            "step_idx": bus.get('step_idx', 'BackendAPI', default=None),
            "performance_data": bus.get('performance_data', 'BackendAPI', default={}) or {},
            "system_health": bus.get('system_health', 'BackendAPI', default=None),
        }
        return {"success": True, **data, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ================== ENHANCED POSITION/TRADING ENDPOINTS ==================
@app.get("/api/position/decisions")
async def position_decisions():
    """Get detailed position decisions with rationale, confidence, and risk factors per instrument"""
    try:
        position_decisions = get_bus_value_with_fallback('position_decisions', 'BackendAPI', default={}) or {}
        portfolio_state = get_bus_value_with_fallback('portfolio_state', 'BackendAPI', default={}) or {}
        current_positions = get_bus_value_with_fallback('current_positions', 'BackendAPI', default={}) or {}
        position_manager_data = get_bus_value_with_fallback('position_manager_data', 'BackendAPI', default={}) or {}
        position_health = get_bus_value_with_fallback('position_health', 'BackendAPI', default={}) or {}
        
        # Build detailed decisions per instrument
        detailed_decisions = {}
        for instrument, decision in position_decisions.items():
            detailed_decisions[instrument] = {
                "decision": decision.get("decision", "hold"),
                "intensity": decision.get("intensity", 0.0),
                "size": decision.get("size", 0.0),
                "confidence": decision.get("confidence", 0.5),
                "risk_factors": decision.get("risk_factors", {}),
                "rationale": decision.get("rationale", []),
                "voting_info": decision.get("voting_info", {}),
                "current_position": current_positions.get(instrument.replace("/", "").replace("_", ""), {}),
            }
        
        return sanitize_for_json({
            "success": True,
            "decisions": detailed_decisions,
            "portfolio_state": {
                "health_score": portfolio_state.get("health_score", 0.0),
                "exposure_ratio": portfolio_state.get("exposure_ratio", 0.0),
                "balance": portfolio_state.get("balance", 0.0),
                "drawdown": portfolio_state.get("drawdown", 0.0),
                "open_positions": portfolio_state.get("open_positions", 0),
            },
            "position_health": position_health,
            "position_count": len(current_positions),
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/voting/breakdown")
async def voting_breakdown():
    """Get individual voter proposals with confidence scores and breakdown"""
    try:
        # Get all voting proposals from individual voters
        voter_proposals = {}
        voter_names = [
            "PPOAgent", "MetaAgent", "DynamicRiskController", "EnhancedAnomalyDetector",
            "ExecutionQualityMonitor", "PortfolioRiskSystem", "EnhancedSeasonalityRiskExpert",
            "EnhancedThemeExpert"
        ]
        
        for voter in voter_names:
            proposal = get_bus_value_with_fallback(f'{voter}_voting_proposal', 'BackendAPI', default=None)
            confidence = get_bus_value_with_fallback(f'{voter}_confidence', 'BackendAPI', default=None)
            if proposal or confidence is not None:
                voter_proposals[voter] = {
                    "proposal": proposal or {},
                    "confidence": confidence if confidence is not None else 0.5,
                    "active": True
                }
        
        # Get voting summary and committee data
        voting_summary = get_bus_value_with_fallback('voting_summary', 'BackendAPI', default={}) or {}
        committee_decision = get_bus_value_with_fallback('committee_decision', 'BackendAPI', default={}) or {}
        committee_votes = get_bus_value_with_fallback('committee_votes', 'BackendAPI', default={}) or {}
        member_confidences = get_bus_value_with_fallback('member_confidences_ordered', 'BackendAPI', default=[]) or []
        consensus_score = get_bus_value_with_fallback('consensus_score', 'BackendAPI', default=0.5)
        agreement_score = get_bus_value_with_fallback('agreement_score', 'BackendAPI', default=0.0)
        
        return sanitize_for_json({
            "success": True,
            "voters": voter_proposals,
            "voter_count": len(voter_proposals),
            "voting_summary": voting_summary,
            "committee_decision": committee_decision,
            "committee_votes": committee_votes,
            "member_confidences": member_confidences,
            "consensus_score": consensus_score,
            "agreement_score": agreement_score,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/strategy/curriculum")
async def strategy_curriculum():
    """Get curriculum progress, competency scores, and learning stage"""
    try:
        curriculum_stage = get_bus_value_with_fallback('curriculum_stage', 'BackendAPI', default={}) or {}
        competency_scores = get_bus_value_with_fallback('competency_scores', 'BackendAPI', default={}) or {}
        learning_recommendations = get_bus_value_with_fallback('learning_recommendations', 'BackendAPI', default=[]) or []
        stage_progression = get_bus_value_with_fallback('stage_progression', 'BackendAPI', default={}) or {}
        mastery_assessment = get_bus_value_with_fallback('mastery_assessment', 'BackendAPI', default={}) or {}
        learning_constraints = get_bus_value_with_fallback('learning_constraints', 'BackendAPI', default={}) or {}
        
        return sanitize_for_json({
            "success": True,
            "current_stage": {
                "index": curriculum_stage.get("stage_index", 0),
                "name": curriculum_stage.get("stage_name", "Unknown"),
                "description": curriculum_stage.get("description", ""),
                "progress": curriculum_stage.get("progress", 0.0),
            },
            "competency_scores": competency_scores,
            "learning_recommendations": learning_recommendations,
            "stage_progression": stage_progression,
            "mastery_assessment": mastery_assessment,
            "learning_constraints": learning_constraints,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/strategy/bias")
async def strategy_bias():
    """Get psychological bias analysis and adjustments"""
    try:
        bias_analysis = get_bus_value_with_fallback('bias_analysis', 'BackendAPI', default={}) or {}
        bias_adjustments = get_bus_value_with_fallback('bias_adjustments', 'BackendAPI', default={}) or {}
        psychological_state = get_bus_value_with_fallback('psychological_state', 'BackendAPI', default={}) or {}
        bias_corrections = get_bus_value_with_fallback('bias_corrections', 'BackendAPI', default={}) or {}
        bias_recommendations = get_bus_value_with_fallback('bias_recommendations', 'BackendAPI', default=[]) or []
        bias_report = get_bus_value_with_fallback('bias_report', 'BackendAPI', default={}) or {}
        
        # Extract aggregate metrics
        aggregate = bias_analysis.get("aggregate_metrics", {})
        individual = bias_analysis.get("individual_biases", {})
        
        return sanitize_for_json({
            "success": True,
            "bias_scores": {
                "total_bias_score": aggregate.get("total_bias_score", 0.0),
                "dominant_bias": aggregate.get("dominant_bias", "none"),
                "bias_severity": aggregate.get("bias_severity", "low"),
            },
            "individual_biases": individual,
            "adjustments": bias_adjustments,
            "psychological_state": psychological_state,
            "corrections": bias_corrections,
            "recommendations": bias_recommendations,
            "report": bias_report,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/strategy/opponent")
async def strategy_opponent():
    """Get opponent simulation data and effectiveness"""
    try:
        opponent_simulation = get_bus_value_with_fallback('opponent_simulation', 'BackendAPI', default={}) or {}
        opponent_analysis = get_bus_value_with_fallback('opponent_analysis', 'BackendAPI', default={}) or {}
        opponent_mode = get_bus_value_with_fallback('opponent_mode', 'BackendAPI', default="random")
        adversarial_scenarios = get_bus_value_with_fallback('adversarial_scenarios', 'BackendAPI', default=[]) or []
        
        # Extract effectiveness metrics
        effectiveness = opponent_analysis.get("effectiveness", {})
        
        return sanitize_for_json({
            "success": True,
            "simulation": {
                "mode": opponent_simulation.get("mode", "random"),
                "intensity": opponent_simulation.get("intensity", 1.0),
                "adaptive_intensity": opponent_simulation.get("adaptive_intensity", 1.0),
                "context_adjustments": opponent_simulation.get("context_adjustments", {}),
            },
            "effectiveness": {
                "score": effectiveness.get("score", 0.5),
                "impact_variance": effectiveness.get("impact_variance", 0.0),
                "robustness_contribution": effectiveness.get("robustness_contribution", 0.0),
            },
            "current_mode": opponent_mode,
            "adversarial_scenarios": adversarial_scenarios[:10],  # Limit to 10
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/strategy/genome")
async def strategy_genome():
    """Get strategy genome pool evolution data"""
    try:
        best_genome = get_bus_value_with_fallback('best_genome', 'BackendAPI', default={}) or {}
        genome_weights = get_bus_value_with_fallback('genome_weights', 'BackendAPI', default={}) or {}
        genome_analysis = get_bus_value_with_fallback('genome_analysis', 'BackendAPI', default={}) or {}
        genome_recommendations = get_bus_value_with_fallback('genome_recommendations', 'BackendAPI', default=[]) or []
        evolution_history = get_bus_value_with_fallback('evolution_history', 'BackendAPI', default=[]) or []
        
        return sanitize_for_json({
            "success": True,
            "best_genome": {
                "parameters": best_genome.get("parameters", []),
                "fitness": best_genome.get("fitness", 0.0),
                "generation": best_genome.get("generation", 0),
            },
            "active_weights": {
                "genome": genome_weights.get("active_genome", []),
                "genome_idx": genome_weights.get("active_genome_idx", 0),
                "fitness": genome_weights.get("active_fitness", 0.0),
                "population_size": genome_weights.get("population_size", 0),
            },
            "analysis": genome_analysis,
            "recommendations": genome_recommendations[:5],
            "evolution_history": evolution_history[-20:],  # Last 20 entries
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/decisions/timeline")
async def decisions_timeline():
    """Get decision timeline with explanations and rationales"""
    try:
        decision_rationales = get_bus_value_with_fallback('decision_rationales', 'BackendAPI', default=[]) or []
        explanation_metrics = get_bus_value_with_fallback('explanation_metrics', 'BackendAPI', default={}) or {}
        active_theses = get_bus_value_with_fallback('active_theses', 'BackendAPI', default={}) or {}
        best_thesis = get_bus_value_with_fallback('best_thesis', 'BackendAPI', default="")
        thesis_evolution = get_bus_value_with_fallback('thesis_evolution', 'BackendAPI', default={}) or {}
        trade_explanation = get_bus_value_with_fallback('trade_explanation', 'BackendAPI', default={}) or {}
        contextual_narratives = get_bus_value_with_fallback('contextual_narratives', 'BackendAPI', default=[]) or []
        
        return sanitize_for_json({
            "success": True,
            "rationales": decision_rationales[-20:],  # Last 20
            "explanation_metrics": {
                "total_trades_audited": explanation_metrics.get("total_trades_audited", 0),
                "high_confidence_trades": explanation_metrics.get("high_confidence_trades", 0),
                "low_confidence_trades": explanation_metrics.get("low_confidence_trades", 0),
                "missing_explanations": explanation_metrics.get("missing_explanations", 0),
            },
            "thesis": {
                "current": best_thesis,
                "active_theses": active_theses,
                "evolution": thesis_evolution,
            },
            "trade_explanation": trade_explanation,
            "narratives": contextual_narratives[-10:],
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/api/logs/{category}")
async def get_logs(category: str, lines: int = Query(default=100, le=10000)):
    """Enhanced log retrieval"""
    log_dirs = {
        "training": "logs/training",
        "risk": "logs/risk",
        "strategy": "logs/strategy",
        "position": "logs/position",
        "simulation": "logs/simulation",
        "system": "logs",
        "evaluation": "logs/evaluation",
        "monitoring": "logs/monitoring",
    }
    
    if category not in log_dirs:
        raise HTTPException(status_code=404, detail=f"Unknown log category: {category}")
    
    log_dir = log_dirs[category]
    log_files = []
    
    if os.path.exists(log_dir):
        for file_path in glob.glob(os.path.join(log_dir, "*.log")):
            stat = os.stat(file_path)
            log_files.append({
                "path": file_path,
                "name": os.path.basename(file_path),
                "size": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    
    # Get content from most recent file
    content = []
    if log_files:
        latest_file = sorted(log_files, key=lambda x: x["modified"], reverse=True)[0]
        try:
            with open(latest_file["path"], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.readlines()[-lines:]
        except Exception as e:
            logger.error(f"Error reading log file: {e}")
    
    return {
        "category": category,
        "files": log_files,
        "content": content,
        "lines_requested": lines,
        "lines_returned": len(content),
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/checkpoints")
async def list_checkpoints():
    """Enhanced checkpoint listing"""
    checkpoint_dir = "checkpoints"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file_path in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
            stat = os.stat(file_path)
            checkpoints.append({
                "name": os.path.basename(file_path),
                "path": file_path,
                "size": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    
    return {
        "checkpoints": sorted(checkpoints, key=lambda x: x["modified"], reverse=True),
        "total_checkpoints": len(checkpoints),
        "total_size_mb": sum(cp["size_mb"] for cp in checkpoints),
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/checkpoints/save")
async def save_checkpoint(name: str = "manual_checkpoint"):
    """Enhanced checkpoint saving"""
    if not state.model_loaded or state.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = f"checkpoints/{name}_{timestamp}.zip"
        state.model.save(checkpoint_path)
        
        # Save metadata
        metadata = {
            "name": name,
            "timestamp": timestamp,
            "system_status": state.system_status,
            "performance": state.performance_metrics,
            "session_id": state.current_session_id,
        }
        
        metadata_path = f"checkpoints/{name}_{timestamp}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        state.add_alert(f"Checkpoint saved: {name}", "success", "checkpoint")
        
        return {
            "success": True,
            "checkpoint": checkpoint_path,
            "metadata": metadata_path,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        error_msg = f"Failed to save checkpoint: {str(e)}"
        state.add_error(error_msg, "checkpoint")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Enhanced model upload"""
    try:
        if not file.filename or not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only .zip model files are accepted")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f"models/uploaded_{timestamp}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        state.add_alert(f"Model uploaded: {file.filename}", "success", "model")
        
        return {
            "success": True,
            "message": "Model uploaded successfully",
            "filename": file.filename,
            "path": file_path,
            "size_mb": round(len(content) / (1024 * 1024), 2),
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        error_msg = f"Failed to upload model: {str(e)}"
        state.add_error(error_msg, "model")
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/data/upload")
async def upload_csv_data(files: List[UploadFile] = File(...)):
    """Upload CSV files for offline training"""
    try:
        data_dir = "data/processed"
        os.makedirs(data_dir, exist_ok=True)
        
        uploaded_files = []
        for file in files:
            if not file.filename or not file.filename.endswith('.csv'):
                continue
                
            file_path = os.path.join(data_dir, file.filename)
            content = await file.read()
            
            with open(file_path, "wb") as f:
                f.write(content)
                
            # Validate CSV
            try:
                df = pd.read_csv(file_path)
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    os.remove(file_path)
                    raise ValueError(f"Missing required columns: {missing_cols}")
                    
                uploaded_files.append({
                    "filename": file.filename,
                    "path": file_path,
                    "rows": len(df),
                    "columns": list(df.columns)
                })
                
            except Exception as e:
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise ValueError(f"Invalid CSV file {file.filename}: {str(e)}")
        
        state.add_alert(f"Uploaded {len(uploaded_files)} CSV files", "success", "data")
        
        return {
            "success": True,
            "uploaded": uploaded_files,
            "message": f"Successfully uploaded {len(uploaded_files)} CSV files"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/data/list")
async def list_csv_data():
    """List available CSV files for offline training"""
    data_dir = "data/processed"
    files = []
    
    if os.path.exists(data_dir):
        for file_path in glob.glob(os.path.join(data_dir, "*.csv")):
            try:
                df = pd.read_csv(file_path, nrows=5)
                stat = os.stat(file_path)
                
                files.append({
                    "filename": os.path.basename(file_path),
                    "path": file_path,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "columns": list(df.columns),
                    "preview_available": True
                })
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
    
    return {
        "files": files,
        "count": len(files),
        "data_dir": data_dir
    }

@app.post("/api/tensorboard/start")
async def start_tensorboard():
    """Enhanced TensorBoard startup"""
    try:
        if state.tensorboard_process and state.tensorboard_process.poll() is None:
            return {
                "success": True, 
                "message": "TensorBoard already running",
                "url": "http://localhost:6006"
            }
        
        log_dir = "logs/tensorboard"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        cmd = [
            sys.executable, "-m", "tensorboard",
            "--logdir", log_dir,
            "--port", "6006",
            "--host", "0.0.0.0",
            "--reload_interval", "30"
        ]
        
        state.tensorboard_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        # Give it time to start
        await asyncio.sleep(3)
        
        if state.tensorboard_process.poll() is None:
            state.add_alert("TensorBoard started", "success", "tensorboard")
            logger.info("TensorBoard started successfully")
            return {
                "success": True,
                "url": "http://localhost:6006",
                "pid": state.tensorboard_process.pid
            }
        else:
            error = ""
            if state.tensorboard_process.stderr is not None:
                error = state.tensorboard_process.stderr.read().decode()
            logger.error(f"TensorBoard failed to start: {error}")
            return {"success": False, "error": error}
            
    except Exception as e:
        error_msg = f"Error starting TensorBoard: {str(e)}"
        state.add_error(error_msg, "tensorboard")
        return {"success": False, "error": error_msg}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Enhanced WebSocket endpoint with better error handling"""
    await websocket.accept()
    state.websocket_connections.append(websocket)
    logger.info(f"[WS] New connection, total: {len(state.websocket_connections)}")

    try:
        # Send initial state
        await broadcast_system_state()

        # Keep connection alive and handle client messages
        while True:
            # Receive a message (or detect disconnect) first
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                # Normal client disconnect (e.g., dev StrictMode unmount)
                logger.debug("[WS] Client disconnected normally")
                break
            except Exception as e:
                # Treat unexpected receive errors as non-fatal and exit loop quietly
                logger.debug(f"WebSocket receive error: {e}")
                break

            # Parse and handle message
            try:
                message = json.loads(data)
            except json.JSONDecodeError as e:
                logger.warning(f"WebSocket message parsing error: {e}")
                continue

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
            elif message.get("type") == "request_update":
                await broadcast_system_state()

    except WebSocketDisconnect:
        # Already handled above; keep silent
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)
            logger.info(f"[WS] Connection removed, total: {len(state.websocket_connections)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": state.get_uptime(),
        "system_status": state.system_status,
        "mt5_connected": state.mt5_connected,
        "model_loaded": state.model_loaded,
        "active_connections": len(state.websocket_connections),
        "error_count": len(state.errors),
        "warning_count": len(state.warnings),
        "session_id": state.current_session_id,
        "version": "3.1.0",
    }

# API documentation
@app.get("/api")
async def api_documentation():
    """Enhanced API documentation"""
    return {
        "name": "AI Trading Dashboard API",
        "version": "3.1.0",
        "description": "Production-ready AI trading system with enhanced training metrics",
        "features": [
            "PPO reinforcement learning with offline/online modes",
            "Real-time training metrics broadcasting",
            "Comprehensive module monitoring",
            "Real-time WebSocket updates",
            "Advanced risk management",
            "Emergency stop controls",
            "Performance analytics",
            "System health monitoring",
            "CSV data management",
        ],
        "endpoints": {
            "authentication": {
                "POST /api/login": "Login to MT5",
                "POST /api/logout": "Logout and cleanup",
            },
            "trading": {
                "POST /api/trading/start": "Start live trading",
                "POST /api/trading/stop": "Stop live trading",
                "POST /api/trading/emergency-stop": "Emergency stop all trading",
            },
            "monitoring": {
                "GET /api/status": "Comprehensive system status",
                "GET /api/modules": "List all modules",
                "GET /api/modules/{module_name}": "Get module details",
                "POST /api/modules/{module_name}/toggle": "Toggle module",
                "GET /api/performance": "Performance metrics",
                "GET /api/alerts": "System alerts",
                "GET /api/logs/{category}": "Get logs",
            },
            "data_management": {
                "POST /api/data/upload": "Upload CSV files",
                "GET /api/data/list": "List CSV files",
            },
            "model_management": {
                "GET /api/checkpoints": "List checkpoints",
                "POST /api/checkpoints/save": "Save checkpoint",
                "POST /api/model/upload": "Upload model",
            },
            "tools": {
                "POST /api/tensorboard/start": "Start TensorBoard",
            },
            "realtime": {
                "WS /ws": "WebSocket for real-time updates",
            },
        },
        "documentation": "/docs",
        "session_id": state.current_session_id,
        "timestamp": datetime.now().isoformat(),
    }

# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# Serve Static Frontend - MUST BE LAST
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info(f"[OK] Frontend served from: {frontend_dist}")
else:
    logger.warning("[WARN] Frontend build not found. Run: cd frontend && npm run build")
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_fallback():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Dashboard - Build Required</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { 
                    background: linear-gradient(135deg, #1e3a8a 0%, #1f2937 50%, #1e3a8a 100%);
                    color: white; 
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    min-height: 100vh;
                    margin: 0;
                    padding: 2rem;
                }
                .container {
                    text-align: center;
                    padding: 3rem;
                    background: rgba(31, 41, 55, 0.9);
                    border-radius: 1rem;
                    border: 1px solid rgba(59, 130, 246, 0.3);
                    backdrop-filter: blur(10px);
                    max-width: 600px;
                    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
                }
                h1 { color: #60a5fa; margin-bottom: 1rem; font-size: 2.5rem; }
                code {
                    background: rgba(17, 24, 39, 0.8);
                    padding: 1rem 1.5rem;
                    border-radius: 0.5rem;
                    display: block;
                    margin: 1rem 0;
                    font-size: 1.1rem;
                    border: 1px solid rgba(59, 130, 246, 0.2);
                }
                .links { margin-top: 2rem; }
                .links a {
                    color: #60a5fa;
                    text-decoration: none;
                    margin: 0 1rem;
                    padding: 0.5rem 1rem;
                    border: 1px solid #60a5fa;
                    border-radius: 0.5rem;
                    transition: all 0.3s ease;
                }
                .links a:hover {
                    background: #60a5fa;
                    color: #1f2937;
                }
                .status {
                    background: rgba(34, 197, 94, 0.1);
                    border: 1px solid rgba(34, 197, 94, 0.3);
                    padding: 1rem;
                    border-radius: 0.5rem;
                    margin: 1rem 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>[BOT] AI Trading Dashboard</h1>
                <div class="status">
                    [OK] Backend is running successfully!
                </div>
                <p>The frontend needs to be built to access the full dashboard.</p>
                <p>To build the frontend, run:</p>
                <code>cd frontend && npm install && npm run build</code>
                <div class="links">
                    <a href="/docs">ðŸ“š API Documentation</a>
                    <a href="/health">ðŸ’š Health Check</a>
                    <a href="/api">[TOOL] API Info</a>
                </div>
            </div>
        </body>
        </html>
        """

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
    reload=True,
    log_level="debug",
        access_log=True
    )
