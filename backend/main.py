#!/usr/bin/env python3
"""
AI Trading System Backend - Live Trading Only
FastAPI server for live MT5 trading with comprehensive monitoring and control.
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import MetaTrader5 as _mt5

mt5: Any = cast(Any, _mt5)

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/backend.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("TradingDashboard")

try:
    from config import get_config as _get_app_config
except Exception:
    _get_app_config = None


def _load_live_app_config():
    if _get_app_config is None:
        return None
    try:
        return _get_app_config(mode="live")
    except Exception:
        return None


_LIVE_APP_CONFIG = _load_live_app_config()


def _live_env_attr(attr: str, default: Any) -> Any:
    if _LIVE_APP_CONFIG and hasattr(_LIVE_APP_CONFIG.environment, attr):
        return getattr(_LIVE_APP_CONFIG.environment, attr)
    return default


def _live_risk_override(attr: str, default: Any) -> Any:
    if _LIVE_APP_CONFIG:
        val = (_LIVE_APP_CONFIG.risk.overrides or {}).get(attr)
        if val is not None:
            return val
    return default


def _live_logging_debug(default: bool = False) -> bool:
    if _LIVE_APP_CONFIG:
        return bool(_LIVE_APP_CONFIG.logging.debug)
    return default


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════


def sanitize_for_json(obj: Any) -> Any:
    """Recursively sanitize an object for JSON serialization."""
    import math

    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(item) for item in obj]
    if isinstance(obj, (deque, set, frozenset)):
        return [sanitize_for_json(item) for item in obj]
    if hasattr(obj, "__dict__"):
        try:
            return sanitize_for_json(vars(obj))
        except Exception:
            return str(obj)
    return obj


INFOBUS_PERSISTENCE_FILE = Path("state/infobus_data.json")
_persisted_cache: Dict[str, Any] = {}
_persisted_cache_time: float = 0.0
_CACHE_TTL_SECONDS: float = 0.5


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

        with open(INFOBUS_PERSISTENCE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        _persisted_cache = {
            k: v.get("value") if isinstance(v, dict) else v for k, v in data.items()
        }
        _persisted_cache_time = now
        return _persisted_cache
    except Exception as e:
        _persisted_cache_time = now
        logger.debug(f"[PERSISTENCE] Cache refresh failed: {e}")
        return _persisted_cache


def get_persisted_bus_value(key: str, default: Any = None) -> Any:
    """Read a value from cached persisted InfoBus data."""
    try:
        cache = _refresh_persisted_cache()
        return cache.get(key, default)
    except Exception:
        return default


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
    return get_persisted_bus_value(key, default)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI App
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="AI Trading Dashboard API",
    version="4.0.0",
    description="Live MT5 trading system with comprehensive monitoring",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════════════════════════════════════


class LoginRequest(BaseModel):
    login: int
    password: str
    server: str = "MetaQuotes-Demo"


class LiveTradingConfig(BaseModel):
    """Live trading configuration.
    
    NOTE: These defaults are conservative fallbacks. The actual values should
    come from risk_policy.yaml via start_live_trading.py which reads:
    - prop_firm.daily_drawdown_limit (5%)
    - prop_firm.max_drawdown_limit (10%)
    - limits.max_position_size
    - limits.max_exposure_pct
    """

    instruments: List[str] = Field(default_factory=lambda: list(_live_env_attr("instruments", ["EURUSD", "XAUUSD"])))
    timeframes: List[str] = Field(default_factory=lambda: list(_live_env_attr("timeframes", ["M15", "H1", "H4", "D1"])))
    update_interval: int = Field(default_factory=lambda: int(_live_env_attr("update_interval", 1)), ge=1, le=60)
    max_position_size: float = Field(default_factory=lambda: float(_live_risk_override("max_position_pct", 0.05)), gt=0, le=1)
    max_total_exposure: float = Field(default_factory=lambda: float(_live_risk_override("max_total_exposure", 0.15)), gt=0, le=1)
    min_trade_interval: int = Field(default_factory=lambda: int(_live_env_attr("min_trade_interval", 60)), ge=10, le=3600)
    use_trailing_stop: bool = Field(default_factory=lambda: bool(_live_env_attr("use_trailing_stop", True)))
    # CRITICAL: Default to conservative 4.2% (below 5% daily limit)
    emergency_drawdown_limit: float = Field(
        default_factory=lambda: float(_live_risk_override("emergency_drawdown_trigger", 0.042)),
        gt=0,
        le=0.5,
    )
    debug: bool = Field(default_factory=lambda: _live_logging_debug(False))


# ══════════════════════════════════════════════════════════════════════════════
# Global State Management
# ══════════════════════════════════════════════════════════════════════════════


class TradingSystemState:
    """State management for live trading system."""

    trading_task: Optional[asyncio.Task[Any]]
    monitoring_tasks: List[asyncio.Task[Any]]
    broadcast_lock: Optional[asyncio.Lock]
    live_env: Optional[Any]
    model: Optional[Any]
    last_trade_time: Dict[str, float]
    trading_config: Optional[LiveTradingConfig]
    websocket_connections: List[WebSocket]
    module_states: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    alerts: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]

    def __init__(self) -> None:
        self.startup_time = datetime.now()
        self.system_status = "IDLE"
        self.mt5_connected = False
        self.model_loaded = False
        self.current_session_id = str(uuid.uuid4())

        self.trading_task = None
        self.monitoring_tasks = []
        self.live_env = None
        self.model = None
        self.last_trade_time = {}
        self.trading_config = None

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
            "trades_today": 0,
            "last_trade_time": None,
        }

        self.module_states = {}
        self.websocket_connections = []
        self.broadcast_lock = None
        self.errors = []
        self.warnings = []
        self.alerts = []
        self.system_metrics = {
            "cpu_usage": 0.0,
            "memory_usage": 0.0,
            "disk_usage": 0.0,
            "network_latency": 0.0,
            "active_connections": 0,
            "last_health_check": datetime.now().isoformat(),
        }

        self._load_modules_from_registry()

    def _load_modules_from_registry(self) -> None:
        """Load module names from YAML registry."""
        try:
            registry_path = "config/module_registry.yaml"
            if not os.path.exists(registry_path):
                return
            with open(registry_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            mods = data.get("modules", {}) or {}
            now = datetime.now().isoformat()
            for name, info in mods.items():
                cat = (info or {}).get("category", "other")
                self.module_states[name] = {
                    "enabled": True,
                    "status": "IDLE",
                    "category": cat.lower() if cat else "other",
                    "last_update": now,
                    "errors": [],
                }
        except Exception as e:
            logger.warning(f"Failed to load module registry: {e}")

    def _sync_modules_from_orchestrator(self) -> bool:
        """Sync module_states from the live ModuleOrchestrator."""
        try:
            from modules.core.module_system import ModuleOrchestrator
        except Exception:
            return False

        try:
            orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()
        except Exception:
            return False

        try:
            mods = getattr(orch, "modules", {}) or {}
            meta = getattr(orch, "metadata", {}) or {}
            perf = getattr(orch, "module_performance", {}) or {}
            bus = getattr(orch, "smart_bus", None)
            now = datetime.now().isoformat()

            new_states: Dict[str, Dict[str, Any]] = {}
            for name in mods.keys():
                m = meta.get(name)
                cat = ""
                try:
                    cat = getattr(m, "category", "") if m is not None else ""
                except Exception:
                    cat = ""
                enabled = True
                try:
                    if bus is not None and hasattr(bus, "is_module_enabled"):
                        enabled = bool(bus.is_module_enabled(name))
                except Exception:
                    pass
                p = perf.get(name, {}) if isinstance(perf, dict) else {}
                status = "idle"
                try:
                    avg_ms = float(p.get("avg_time_ms", 0.0) or 0.0)
                    err_rate = float(p.get("error_rate", 0.0) or 0.0)
                    if avg_ms > 0.0:
                        status = "monitoring" if err_rate < 0.5 else "error"
                except Exception:
                    pass

                new_states[name] = {
                    "enabled": enabled,
                    "status": status,
                    "category": (cat or "other").lower(),
                    "last_update": now,
                    "errors": [],
                }

            if new_states:
                self.module_states = new_states
                return True
            return False
        except Exception:
            return False

    def get_uptime(self) -> str:
        """Get system uptime."""
        uptime = datetime.now() - self.startup_time
        days = uptime.days
        hours, remainder = divmod(uptime.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if days > 0:
            return f"{days}d {hours}h {minutes}m"
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {seconds}s"

    def add_error(self, error: str, module: str = "system") -> None:
        """Add error with tracking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "error": error,
            "severity": "error",
            "session_id": self.current_session_id,
        }
        self.errors.append(entry)
        self.errors = self.errors[-1000:]

        if module in self.module_states:
            if "errors" not in self.module_states[module]:
                self.module_states[module]["errors"] = []
            self.module_states[module]["errors"].append(error)
            self.module_states[module]["errors"] = self.module_states[module][
                "errors"
            ][-10:]

    def add_warning(self, warning: str, module: str = "system") -> None:
        """Add warning with tracking."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "warning": warning,
            "severity": "warning",
            "session_id": self.current_session_id,
        }
        self.warnings.append(entry)
        self.warnings = self.warnings[-1000:]

    def add_alert(
        self, alert: str, severity: str = "info", module: str = "system"
    ) -> None:
        """Add system alert."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "module": module,
            "alert": alert,
            "severity": severity,
            "session_id": self.current_session_id,
        }
        self.alerts.append(entry)
        self.alerts = self.alerts[-500:]


state = TradingSystemState()


# ══════════════════════════════════════════════════════════════════════════════
# MT5 Integration
# ══════════════════════════════════════════════════════════════════════════════


def connect_mt5(login: int, password: str, server: str) -> Dict[str, Any]:
    """Connect to MT5 broker."""
    try:
        logger.info(f"Attempting MT5 connection - Login: {login}, Server: {server}")

        if not mt5.initialize():
            error_code = mt5.last_error()
            error_msg = f"MT5 initialization failed: {error_code}"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}

        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            error_code = mt5.last_error()
            mt5.shutdown()
            error_msg = f"MT5 login failed: {error_code}"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}

        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            error_msg = "Failed to retrieve MT5 account information"
            state.add_error(error_msg, "mt5")
            return {"success": False, "error": error_msg}

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
            },
        }

    except Exception as e:
        error_msg = f"MT5 connection error: {e}"
        state.add_error(error_msg, "mt5")
        logger.error(error_msg)
        return {"success": False, "error": error_msg}


def disconnect_mt5() -> None:
    """Disconnect from MT5."""
    try:
        if state.mt5_connected:
            mt5.shutdown()
            state.mt5_connected = False
            logger.info("MT5 disconnected successfully")
    except Exception as e:
        error_msg = f"MT5 disconnection error: {e}"
        state.add_error(error_msg, "mt5")
        logger.error(error_msg)


# ══════════════════════════════════════════════════════════════════════════════
# Live Trading System
# ══════════════════════════════════════════════════════════════════════════════


async def start_live_trading(config: LiveTradingConfig) -> Dict[str, Any]:
    """Start live trading with the PPO model."""
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")

        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Trading already active")

        # Set trading mode to LIVE
        try:
            from modules.core.trading_mode import TradingModeManager

            TradingModeManager.set_mode("LIVE")
            logger.info("[LIVE MODE] TradingModeManager set to LIVE")
        except Exception as e:
            logger.warning(f"Failed to set TradingModeManager to LIVE: {e}")

        # Set environment config on InfoBus
        try:
            from modules.utils.info_bus import InfoBusManager

            bus = InfoBusManager.get_instance()

            def _norm_inst(s: str) -> str:
                try:
                    s = str(s)
                    if "/" in s:
                        return s
                    if "_" in s:
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
                "mode": "live",
                "max_steps": 100000,
                "bus_data_active": True,
            }

            bus.set(
                "environment_config",
                environment_config,
                module="Backend",
                thesis="live trading environment configuration",
            )
            bus.set(
                "execution_mode",
                "live",
                module="Backend",
                thesis="live trading mode enabled",
            )
            logger.info(f"[LIVE MODE] Set environment config on InfoBus")
        except Exception as e:
            logger.error(f"Failed to set environment config on InfoBus: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to configure live mode: {e}"
            )

        # Load PPO model
        model_path = "models/ppo_trading_model.zip"
        if not os.path.exists(model_path):
            alt_paths = ["models/ppo_final_model.zip", "models/modern_ppo_final.zip"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail="PPO model not found")

        logger.info(f"Starting live trading system with model: {model_path}")

        from stable_baselines3 import PPO

        from envs.env import EnhancedTradingEnv, TradingConfig

        try:
            from live.live_connector import LiveDataConnector
        except Exception:
            from live.live_connector import InfoBusLiveDataConnector as LiveDataConnector

        connector = LiveDataConnector(
            instruments=config.instruments, timeframes=config.timeframes
        )
        connector.connect()

        if hasattr(connector, "get_historical_data"):
            hist_data = connector.get_historical_data(n_bars=1000)
        elif hasattr(connector, "get_historical_data_with_infobus"):
            hist_data = connector.get_historical_data_with_infobus(n_bars=1000)
        else:
            raise HTTPException(
                status_code=500,
                detail="Connector does not support historical data retrieval",
            )
        if not hist_data:
            raise HTTPException(
                status_code=500, detail="Failed to retrieve historical data"
            )

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

        state.trading_task = asyncio.create_task(live_trading_loop(config, connector))
        state.system_status = "TRADING"
        state.add_alert("Live trading started successfully", "success", "trading")
        logger.info("Live trading started successfully")

        await broadcast_system_state()

        return {
            "success": True,
            "message": "Live trading started",
            "session_id": state.current_session_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to start live trading: {e}"
        state.add_error(error_msg, "trading")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


async def live_trading_loop(config: LiveTradingConfig, connector: Any) -> None:
    """Main live trading loop."""
    try:
        assert state.live_env is not None
        assert state.model is not None
        obs, _ = state.live_env.reset()
        last_balance_update = time.time()
        last_health_check = time.time()
        last_state_save = time.time()
        STATE_SAVE_INTERVAL = 300

        orchestrator = None
        try:
            from modules.core.module_system import ModuleOrchestrator

            orchestrator = ModuleOrchestrator.get_instance()
            await asyncio.get_event_loop().run_in_executor(
                None, orchestrator.initialize
            )
            logger.info("ModuleOrchestrator initialized for live trading")
        except Exception as e:
            logger.warning(f"Failed to initialize ModuleOrchestrator: {e}")

        logger.info("Live trading loop started")

        while state.system_status == "TRADING":
            loop_start = time.time()

            try:
                new_data = connector.get_historical_data(n_bars=1)
                if new_data:
                    _update_environment_data(new_data, config)

                if orchestrator:
                    try:
                        market_data = new_data if new_data else {}
                        await orchestrator.execute_step(market_data)
                        state._sync_modules_from_orchestrator()
                    except Exception as e:
                        logger.warning(f"Module execution error: {e}")

                action, _ = state.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = state.live_env.step(action)
                _update_module_states(info)

                if time.time() - last_balance_update > 30:
                    _update_balance_from_broker()
                    _sync_trade_performance_from_bus()  # CRITICAL: Sync accurate trade metrics
                    last_balance_update = time.time()

                if time.time() - last_health_check > 60:
                    _perform_health_checks()
                    last_health_check = time.time()

                if orchestrator and time.time() - last_state_save > STATE_SAVE_INTERVAL:
                    try:
                        results = orchestrator.state_manager.save_all_module_states(
                            orchestrator
                        )
                        saved = sum(1 for ok in results.values() if ok)
                        logger.info(
                            f"[SAVE] Periodic state save: {saved}/{len(results)} modules saved"
                        )
                        last_state_save = time.time()
                    except Exception as e:
                        logger.warning(f"Periodic state save failed: {e}")

                if _check_emergency_conditions():
                    logger.warning("Emergency conditions detected, stopping trading")
                    await emergency_stop()
                    break

                await broadcast_system_state()

                loop_time = time.time() - loop_start
                sleep_time = max(0, config.update_interval - loop_time)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                error_msg = f"Trading loop step error: {e}"
                state.add_error(error_msg, "trading")
                logger.error(error_msg)
                await asyncio.sleep(config.update_interval)

    except Exception as e:
        error_msg = f"Trading loop fatal error: {e}"
        state.add_error(error_msg, "trading")
        logger.error(error_msg)
        state.system_status = "ERROR"
    finally:
        connector.disconnect()
        logger.info("Live trading loop ended")


def _update_environment_data(new_data: Dict[str, Any], config: LiveTradingConfig) -> None:
    """Update environment with new market data."""
    try:
        for inst in config.instruments:
            inst_key = inst[:3] + "/" + inst[3:] if len(inst) == 6 else inst
            for tf in config.timeframes:
                if inst_key in new_data and tf in new_data[inst_key]:
                    if (
                        state.live_env is not None
                        and hasattr(state.live_env, "data")
                        and inst_key in state.live_env.data
                    ):
                        state.live_env.data[inst_key][tf] = pd.concat(
                            [
                                state.live_env.data[inst_key][tf].iloc[1:],
                                new_data[inst_key][tf].iloc[-1:],
                            ]
                        )
    except Exception as e:
        state.add_error(f"Data update error: {e}", "data")


def _update_module_states(info: Dict[str, Any]) -> None:
    """Update module states from environment info."""
    try:
        current_time = datetime.now().isoformat()

        if "position_manager" in info:
            pm_info = info["position_manager"]
            state.module_states.setdefault("position_manager", {}).update(
                {
                    "status": "active",
                    "last_update": current_time,
                    "open_positions": pm_info.get("open_positions", {}),
                    "total_exposure": pm_info.get("total_exposure", 0.0),
                    "position_count": pm_info.get("position_count", 0),
                }
            )

        if "risk" in info:
            risk_info = info["risk"]
            state.module_states.setdefault("risk_controller", {}).update(
                {
                    "status": "monitoring",
                    "last_update": current_time,
                    "risk_scale": risk_info.get("risk_scale", 1.0),
                    "risk_level": risk_info.get("risk_level", "NORMAL"),
                    "drawdown": risk_info.get("drawdown", 0.0),
                }
            )

        if "votes" in info:
            vote_info = info["votes"]
            state.module_states.setdefault("strategy_arbiter", {}).update(
                {
                    "status": "voting",
                    "last_update": current_time,
                    "consensus": vote_info.get("consensus", 0.0),
                    "gate_status": vote_info.get("gate_status", "OPEN"),
                }
            )

    except Exception as e:
        state.add_error(f"Module state update error: {e}", "system")


def _update_balance_from_broker() -> None:
    """Update balance from MT5 broker."""
    try:
        if state.mt5_connected:
            account_info = mt5.account_info()
            if account_info:
                new_balance = account_info.balance
                state.performance_metrics["current_balance"] = new_balance
                state.performance_metrics["daily_pnl"] = (
                    new_balance - state.performance_metrics["start_balance"]
                )
                state.performance_metrics["total_pnl"] = (
                    new_balance - state.performance_metrics["start_balance"]
                )

                if new_balance > state.performance_metrics["peak_balance"]:
                    state.performance_metrics["peak_balance"] = new_balance

                peak = state.performance_metrics["peak_balance"]
                current_dd = (peak - new_balance) / peak if peak > 0 else 0.0
                state.performance_metrics["current_drawdown"] = current_dd

                if current_dd > state.performance_metrics["max_drawdown"]:
                    state.performance_metrics["max_drawdown"] = current_dd

    except Exception as e:
        state.add_error(f"Balance update error: {e}", "mt5")


def _sync_trade_performance_from_bus() -> None:
    """
    Sync trade performance metrics from InfoBus closed_positions.
    
    CRITICAL FIX: Use closed_positions for accurate win rate calculation.
    The recent_trades/trades keys contain ALL fills (opens + closes) which double-counts.
    closed_positions contains only completed round-trip trades.
    """
    try:
        closed_positions = get_bus_value_with_fallback("closed_positions", "BackendAPI", default=[]) or []
        
        if not closed_positions:
            # No closed positions yet - don't overwrite metrics
            return
        
        total_trades = len(closed_positions)
        winning_trades = sum(1 for t in closed_positions if (t.get("pnl", 0) or t.get("profit", 0) or 0) > 0)
        losing_trades = sum(1 for t in closed_positions if (t.get("pnl", 0) or t.get("profit", 0) or 0) < 0)
        
        # Calculate win rate from actual closed trades
        win_rate = winning_trades / max(1, total_trades)
        
        # Update state metrics
        state.performance_metrics["total_trades"] = total_trades
        state.performance_metrics["winning_trades"] = winning_trades
        state.performance_metrics["losing_trades"] = losing_trades
        state.performance_metrics["win_rate"] = win_rate
        
        # Calculate profit factor from closed positions
        total_wins = sum(float(t.get("pnl", 0) or t.get("profit", 0) or 0) for t in closed_positions if (t.get("pnl", 0) or t.get("profit", 0) or 0) > 0)
        total_losses = abs(sum(float(t.get("pnl", 0) or t.get("profit", 0) or 0) for t in closed_positions if (t.get("pnl", 0) or t.get("profit", 0) or 0) < 0))
        
        if total_losses > 0:
            state.performance_metrics["profit_factor"] = total_wins / total_losses
        elif total_wins > 0:
            state.performance_metrics["profit_factor"] = 999.0  # All wins, no losses
        else:
            state.performance_metrics["profit_factor"] = 1.0  # No trades with P&L
            
    except Exception as e:
        logger.debug(f"Trade performance sync from bus failed (may be normal during startup): {e}")


def _perform_health_checks() -> None:
    """Perform comprehensive system health checks."""
    try:
        if state.mt5_connected:
            terminal_info = mt5.terminal_info()
            if not terminal_info or not terminal_info.trade_allowed:
                state.add_warning("MT5 trading not allowed", "mt5")

            try:
                positions = mt5.positions_get()
                if positions:
                    missing_sl_count = sum(1 for p in positions if p.sl <= 0)
                    if missing_sl_count > 0:
                        state.add_warning(
                            f"⚠️ {missing_sl_count} positions without Stop Loss!", "risk"
                        )
                        logger.warning(
                            f"[RISK] {missing_sl_count} positions without Stop Loss"
                        )
                        asyncio.create_task(_auto_fix_sl_tp())
            except Exception as e:
                logger.error(f"Error checking positions SL/TP: {e}")

        if state.model_loaded and state.model is None:
            state.add_error("Model loaded flag set but model is None", "model")
            state.model_loaded = False

        import psutil

        memory_percent = psutil.virtual_memory().percent
        if memory_percent > 90:
            state.add_warning(f"High memory usage: {memory_percent:.1f}%", "system")

        state.system_metrics.update(
            {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": memory_percent,
                "disk_usage": psutil.disk_usage("/").percent,
                "last_health_check": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        state.add_error(f"Health check error: {e}", "system")


async def _auto_fix_sl_tp() -> None:
    """Auto-fix positions missing SL/TP."""
    try:
        try:
            if _LIVE_APP_CONFIG:
                sl_tp_config = (_LIVE_APP_CONFIG.risk.policy or {}).get("sl_tp_settings", {}) or {}
            else:
                with open("config/risk_policy.yaml", "r", encoding="utf-8") as f:
                    risk_config = yaml.safe_load(f) or {}
                    sl_tp_config = risk_config.get("sl_tp_settings", {})
        except Exception:
            sl_tp_config = {"auto_sl_enabled": True, "auto_tp_enabled": True}

        if not sl_tp_config.get("fix_missing_sl_tp", True):
            return

        positions = mt5.positions_get()
        if not positions:
            return

        def pips_to_price(symbol: str, pips: float) -> float:
            sym_upper = symbol.upper()
            if "XAU" in sym_upper or "GOLD" in sym_upper:
                return pips * 0.01
            if "JPY" in sym_upper:
                return pips * 0.01
            return pips * 0.0001

        fixed_count = 0
        for pos in positions:
            needs_sl = pos.sl <= 0 and sl_tp_config.get("auto_sl_enabled", True)
            needs_tp = pos.tp <= 0 and sl_tp_config.get("auto_tp_enabled", True)

            if not needs_sl and not needs_tp:
                continue

            symbol = pos.symbol
            _sym_cfg = sl_tp_config.get(symbol, sl_tp_config.get("default", {}))
            symbol_config: Dict[str, Any] = _sym_cfg if isinstance(_sym_cfg, dict) else {}
            sl_pips = symbol_config.get("stop_loss_pips", 50)
            tp_pips = symbol_config.get("take_profit_pips", 100)

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
                logger.info(
                    f"[AUTO-SL/TP] Fixed position {pos.ticket}: SL={new_sl:.5f} TP={new_tp:.5f}"
                )

        if fixed_count > 0:
            logger.info(f"[AUTO-SL/TP] Auto-fixed {fixed_count} positions")

    except Exception as e:
        logger.error(f"[AUTO-SL/TP] Error: {e}")


def _check_emergency_conditions() -> bool:
    """Check for emergency conditions."""
    try:
        current_dd = state.performance_metrics.get("current_drawdown", 0.0)
        max_dd_limit = (
            state.trading_config.emergency_drawdown_limit
            if state.trading_config
            else 0.25
        )

        if current_dd > max_dd_limit:
            state.add_alert(
                f"Emergency: Drawdown {current_dd:.1%} exceeds limit {max_dd_limit:.1%}",
                "critical",
                "risk",
            )
            return True

        risk_state = state.module_states.get("risk_controller", {})
        if risk_state.get("freeze_counter", 0) > 10:
            state.add_alert("Emergency: Risk system frozen too long", "critical", "risk")
            return True

        if len(state.errors) > 100:
            state.add_alert("Emergency: Too many system errors", "critical", "system")
            return True

        return False

    except Exception as e:
        state.add_error(f"Emergency check error: {e}", "system")
        return False


async def emergency_stop() -> Dict[str, Any]:
    """Emergency stop with comprehensive cleanup."""
    try:
        logger.warning("[ALERT] EMERGENCY STOP INITIATED")
        state.add_alert("Emergency stop initiated", "critical", "emergency")

        if state.mt5_connected:
            positions = mt5.positions_get()
            if positions:
                logger.info(f"Closing {len(positions)} open positions...")

                for position in positions:
                    try:
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
                            logger.error(
                                f"[FAIL] Failed to close position {position.ticket}: {result.comment}"
                            )

                    except Exception as e:
                        logger.error(f"Error closing position {position.ticket}: {e}")

        if state.trading_task and not state.trading_task.done():
            state.trading_task.cancel()
            try:
                await state.trading_task
            except asyncio.CancelledError:
                pass

        try:
            from modules.core.module_system import ModuleOrchestrator

            orchestrator = ModuleOrchestrator.get_instance()
            if orchestrator and hasattr(orchestrator, "state_manager"):
                results = orchestrator.state_manager.save_all_module_states(orchestrator)
                saved = sum(1 for ok in results.values() if ok)
                logger.info(
                    f"[SAVE] Emergency shutdown state save: {saved}/{len(results)} modules saved"
                )
        except Exception as e:
            logger.warning(f"Failed to save module states on emergency stop: {e}")

        state.system_status = "EMERGENCY_STOPPED"
        state.add_alert("Emergency stop completed", "warning", "emergency")
        logger.warning("[STOP] Emergency stop completed")

        return {"success": True, "message": "Emergency stop executed"}

    except Exception as e:
        error_msg = f"Emergency stop error: {e}"
        state.add_error(error_msg, "emergency")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Management
# ══════════════════════════════════════════════════════════════════════════════


async def broadcast_system_state() -> None:
    """Broadcast system state to all WebSocket connections."""
    if not state.websocket_connections:
        return

    try:
        regime_analytics: Dict[str, Any] = {}
        try:
            from modules.utils.info_bus import InfoBusManager

            bus = InfoBusManager.get_instance()
            regime_pred = bus.get("regime_prediction", "BackendAPI", default=None)
            regime_probs = bus.get("regime_probabilities", "BackendAPI", default={}) or {}

            def _norm_regime(val: Any) -> Any:
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    for k in ("label", "name", "regime", "state"):
                        if k in val and isinstance(val[k], str):
                            return val[k]
                return val

            regime_analytics = {
                "current_regime": _norm_regime(regime_pred),
                "regime_probabilities": regime_probs,
            }
        except Exception:
            pass

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
                "regime_analytics": regime_analytics,
                "timestamp": datetime.now().isoformat(),
            },
        }

        system_state = sanitize_for_json(system_state)
        await _send_to_all_websockets(system_state)

    except Exception as e:
        logger.error(f"[WS] Broadcast system_state error: {e}")
        state.add_error(f"Broadcast error: {e}", "websocket")


async def broadcast_mt5_data_update() -> None:
    """Broadcast MT5 data updates."""
    if not state.websocket_connections:
        return

    try:
        positions: List[Dict[str, Any]] = []
        account_info: Dict[str, Any] = {}
        symbols: List[Dict[str, Any]] = []

        if state.mt5_connected:
            try:
                acc = mt5.account_info()
                if acc:
                    account_info = {
                        "balance": float(acc.balance),
                        "equity": float(acc.equity),
                        "profit": float(acc.profit),
                        "margin": float(acc.margin),
                        "margin_free": float(acc.margin_free),
                    }
                    state.performance_metrics["current_balance"] = float(acc.balance)
                    state.performance_metrics["total_pnl"] = float(acc.profit)

                mt5_positions = mt5.positions_get()
                if mt5_positions:
                    for pos in mt5_positions:
                        positions.append(
                            {
                                "ticket": pos.ticket,
                                "symbol": pos.symbol,
                                "type": "BUY"
                                if pos.type == mt5.ORDER_TYPE_BUY
                                else "SELL",
                                "volume": float(pos.volume),
                                "price_open": float(pos.price_open),
                                "price_current": float(pos.price_current),
                                "profit": float(pos.profit),
                                "sl": float(pos.sl) if pos.sl else None,
                                "tp": float(pos.tp) if pos.tp else None,
                                "time": datetime.fromtimestamp(pos.time).isoformat(),
                            }
                        )

                for sym in ("EURUSD", "XAUUSD"):
                    try:
                        info = mt5.symbol_info(sym)
                        if info is not None:
                            symbols.append(
                                {
                                    "symbol": getattr(info, "name", sym) or sym,
                                    "bid": float(getattr(info, "bid", 0.0) or 0.0),
                                    "ask": float(getattr(info, "ask", 0.0) or 0.0),
                                    "spread": int(getattr(info, "spread", 0) or 0),
                                }
                            )
                    except Exception:
                        pass
            except Exception as mt5_error:
                logger.warning(f"Could not get live MT5 data: {mt5_error}")

        message = {
            "type": "mt5_data_update",
            "data": {
                "symbols": symbols,
                "positions": positions,
                "account": account_info,
                "positionCount": len(positions),
                "timestamp": datetime.now().isoformat(),
            },
        }

        await _send_to_all_websockets(message)
    except Exception as e:
        logger.error(f"Error broadcasting MT5 data update: {e}")


async def start_real_time_updates() -> None:
    """Start periodic real-time updates."""
    logger.info("[WS] Starting real-time update loop")
    while True:
        try:
            await broadcast_system_state()
            await asyncio.sleep(5)
            await broadcast_mt5_data_update()
            await asyncio.sleep(10)
        except Exception as e:
            logger.error(f"Error in real-time updates: {e}")
            await asyncio.sleep(10)


async def _send_to_all_websockets(message: Dict[str, Any]) -> None:
    """Send message to all connected websockets."""
    lock = getattr(state, "broadcast_lock", None)

    async def _do_send() -> None:
        disconnected = []
        for websocket in list(state.websocket_connections):
            try:
                if hasattr(websocket, "client_state"):
                    from starlette.websockets import WebSocketState

                    if websocket.client_state != WebSocketState.CONNECTED:
                        disconnected.append(websocket)
                        continue
                await websocket.send_json(message)
            except Exception as e:
                error_str = str(e).lower()
                if any(
                    x in error_str
                    for x in ["closed", "disconnect", "connection", "broken pipe"]
                ):
                    disconnected.append(websocket)
                else:
                    logger.warning(f"[WS] Error sending to websocket: {e}")
        for ws in disconnected:
            if ws in state.websocket_connections:
                state.websocket_connections.remove(ws)

    if lock is not None:
        async with lock:
            await _do_send()
    else:
        await _do_send()


# ══════════════════════════════════════════════════════════════════════════════
# Application Events
# ══════════════════════════════════════════════════════════════════════════════


@app.on_event("startup")
async def startup_event() -> None:
    """System startup."""
    directories = [
        "logs",
        "logs/risk",
        "logs/strategy",
        "logs/position",
        "logs/monitoring",
        "models",
        "models/best",
        "state",
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    try:
        from modules.utils.info_bus import InfoBusManager

        bus = InfoBusManager.get_instance()
        default_env_config = {
            "instruments": [],
            "initial_balance": 100000.0,
            "mode": "live",
            "max_steps": 100000,
            "bus_data_active": False,
        }
        bus.set(
            "environment_config",
            default_env_config,
            module="Backend",
            thesis="backend startup - live standby mode",
        )
        bus.set(
            "execution_mode", "live", module="Backend", thesis="backend startup - live mode"
        )
        logger.info("[INIT] InfoBus initialized in LIVE standby mode")
    except Exception as e:
        logger.error(f"Failed to initialize InfoBus: {e}")

    try:
        state.broadcast_lock = asyncio.Lock()
    except Exception:
        state.broadcast_lock = None

    state.monitoring_tasks.extend(
        [
            asyncio.create_task(_periodic_metrics_collector()),
            asyncio.create_task(_system_health_monitor()),
            asyncio.create_task(_performance_tracker()),
            asyncio.create_task(start_real_time_updates()),
        ]
    )

    logger.info("[ROCKET] Trading Dashboard Backend Started")
    state.add_alert("System started successfully", "success", "system")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """System shutdown."""
    logger.info("[STOP] Shutting down trading dashboard...")

    for task in state.monitoring_tasks:
        if not task.done():
            task.cancel()

    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        state.trading_task.cancel()
        try:
            await state.trading_task
        except asyncio.CancelledError:
            pass

    disconnect_mt5()
    logger.info("[OK] Trading Dashboard Backend Shutdown Complete")


async def _periodic_metrics_collector() -> None:
    """Collect system metrics periodically."""
    while True:
        try:
            await broadcast_system_state()
            await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(10)


async def _system_health_monitor() -> None:
    """Monitor system health periodically."""
    while True:
        try:
            _perform_health_checks()
            await asyncio.sleep(60)
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
            await asyncio.sleep(60)


async def _performance_tracker() -> None:
    """Track and update performance metrics."""
    while True:
        try:
            try:
                state._sync_modules_from_orchestrator()
            except Exception:
                pass
            if state.mt5_connected and state.system_status == "TRADING":
                _update_balance_from_broker()
            # Always sync trade performance from bus (works for both sim and live)
            _sync_trade_performance_from_bus()
            await asyncio.sleep(30)
        except Exception as e:
            logger.error(f"Performance tracking error: {e}")
            await asyncio.sleep(30)


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Authentication
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/api/login")
async def login(request: LoginRequest) -> Dict[str, Any]:
    """MT5 login."""
    result = connect_mt5(request.login, request.password, request.server)
    if result["success"]:
        await broadcast_system_state()
        return result
    raise HTTPException(status_code=401, detail=result["error"])


@app.post("/api/logout")
async def logout() -> Dict[str, Any]:
    """Logout with cleanup."""
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


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Trading
# ══════════════════════════════════════════════════════════════════════════════


@app.post("/api/trading/start")
async def trading_start(config: LiveTradingConfig) -> Dict[str, Any]:
    """Start live trading."""
    result = await start_live_trading(config)
    await broadcast_system_state()
    return result


@app.post("/api/trading/stop")
async def trading_stop() -> Dict[str, Any]:
    """Stop live trading."""
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        state.trading_task.cancel()
        try:
            await state.trading_task
        except asyncio.CancelledError:
            pass

        state.system_status = "IDLE"
        state.add_alert("Trading stopped by user", "info", "trading")
        await broadcast_system_state()
        return {"success": True}
    raise HTTPException(status_code=400, detail="No trading process running")


@app.post("/api/trading/emergency-stop")
async def emergency_stop_endpoint() -> Dict[str, Any]:
    """Emergency stop endpoint."""
    result = await emergency_stop()
    await broadcast_system_state()
    return result


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Status & Monitoring
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/status")
async def get_comprehensive_status() -> Dict[str, Any]:
    """Get comprehensive system status."""
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
        "modules": {
            name: {
                "enabled": module.get("enabled", False),
                "status": module.get("status", "unknown"),
                "last_update": module.get("last_update", "never"),
            }
            for name, module in state.module_states.items()
        },
        "health": {
            "errors_count": len(state.errors),
            "warnings_count": len(state.warnings),
            "alerts_count": len(state.alerts),
            "last_health_check": state.system_metrics.get("last_health_check"),
        },
        "system_metrics": state.system_metrics,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/performance")
async def get_performance_metrics() -> Dict[str, Any]:
    """Get detailed performance metrics."""
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
async def get_alerts(limit: int = Query(default=50, le=1000)) -> Dict[str, Any]:
    """Get system alerts."""
    return {
        "alerts": state.alerts[-limit:],
        "total_alerts": len(state.alerts),
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Modules
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/modules")
async def list_modules() -> Dict[str, Any]:
    """Get module states."""
    try:
        modules_data = []

        try:
            from modules.utils.info_bus import InfoBusManager

            bus = InfoBusManager.get_instance()
        except Exception:
            bus = None

        module_registry: Dict[str, Any] = {}
        try:
            with open("config/module_registry.yaml", "r", encoding="utf-8") as f:
                registry_data = yaml.safe_load(f)
                module_registry = registry_data.get("modules", {})
        except Exception:
            pass

        for name, module in state.module_states.items():
            registry_info = module_registry.get(name, {})
            live_data: Dict[str, Any] = {}

            if bus:
                try:
                    for key in registry_info.get("provides", []):
                        value = bus.get(key, name, default=None)
                        if value is not None:
                            live_data[key] = value
                except Exception:
                    pass

            module_data = {
                "name": name,
                "enabled": module.get("enabled", False),
                "status": module.get("status", "unknown").upper(),
                "category": module.get("category", "unknown"),
                "last_update": module.get("last_update", "never"),
                "provides": registry_info.get("provides", []),
                "requires": registry_info.get("requires", []),
                "live_data": live_data,
                "error_count": len(module.get("errors", [])),
                "errors": module.get("errors", [])[-5:],
                "data_richness": len(live_data),
            }
            modules_data.append(module_data)

        modules_data.sort(key=lambda x: (x["category"], x["name"]))

        categories: Dict[str, Dict[str, int]] = {}
        for mod in modules_data:
            cat = mod["category"]
            if cat not in categories:
                categories[cat] = {"total": 0, "enabled": 0, "with_data": 0}
            categories[cat]["total"] += 1
            if mod["enabled"]:
                categories[cat]["enabled"] += 1
            if mod["data_richness"] > 0:
                categories[cat]["with_data"] += 1

        return sanitize_for_json(
            {
                "modules": modules_data,
                "total_modules": len(modules_data),
                "enabled_modules": sum(1 for m in modules_data if m["enabled"]),
                "categories": categories,
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Error in list_modules: {e}", exc_info=True)
        return {
            "modules": [],
            "total_modules": 0,
            "enabled_modules": 0,
            "categories": {},
            "timestamp": datetime.now().isoformat(),
            "error": str(e),
        }


@app.get("/api/modules/{module_name}")
async def get_module_state(module_name: str) -> Dict[str, Any]:
    """Get detailed state for specific module."""
    if module_name not in state.module_states:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")

    return {
        "module": module_name,
        "state": state.module_states[module_name],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/modules/{module_name}/toggle")
async def toggle_module(module_name: str) -> Dict[str, Any]:
    """Toggle module enabled/disabled state."""
    try:
        from modules.core.module_system import ModuleOrchestrator

        orch = ModuleOrchestrator._instance or ModuleOrchestrator.get_instance()
        current_state = bool(
            state.module_states.get(module_name, {}).get("enabled", True)
        )
        if current_state:
            orch.disable_module(module_name, reason="User toggle")
        else:
            orch.enable_module(module_name)
    except Exception:
        pass

    if not state._sync_modules_from_orchestrator():
        cur = state.module_states.get(module_name, {}).get("enabled", False)
        state.module_states.setdefault(module_name, {})["enabled"] = not cur

    action = (
        "enabled"
        if state.module_states.get(module_name, {}).get("enabled", False)
        else "disabled"
    )
    state.add_alert(f"Module {module_name} {action}", "info", module_name)
    await broadcast_system_state()

    return {
        "module": module_name,
        "enabled": state.module_states[module_name]["enabled"],
        "message": f"Module {action} successfully",
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - MT5
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/mt5/status")
async def mt5_status() -> Dict[str, Any]:
    """Get MT5 connection status."""
    info: Dict[str, Any] = {"connected": bool(state.mt5_connected)}
    if state.mt5_connected:
        try:
            ti = mt5.terminal_info()
            if ti is not None:
                info.update(
                    {
                        "trade_allowed": bool(getattr(ti, "trade_allowed", False)),
                        "name": getattr(ti, "name", None),
                        "company": getattr(ti, "company", None),
                    }
                )
        except Exception:
            pass
    return {"success": True, **info, "timestamp": datetime.now().isoformat()}


@app.get("/api/mt5/account")
async def mt5_account() -> Dict[str, Any]:
    """Get MT5 account info."""
    if not state.mt5_connected:
        raise HTTPException(status_code=400, detail="MT5 not connected")
    ai = mt5.account_info()
    if ai is None:
        raise HTTPException(
            status_code=500, detail="Failed to retrieve MT5 account information"
        )
    data = {
        "login": getattr(ai, "login", None),
        "balance": getattr(ai, "balance", None),
        "equity": getattr(ai, "equity", None),
        "margin": getattr(ai, "margin", None),
        "margin_free": getattr(ai, "margin_free", None),
        "currency": getattr(ai, "currency", None),
        "leverage": getattr(ai, "leverage", None),
        "profit": getattr(ai, "profit", None),
        "margin_level": getattr(ai, "margin_level", None),
        "company": getattr(ai, "company", None),
    }
    return {"success": True, "account": data, "timestamp": datetime.now().isoformat()}


@app.get("/api/mt5/positions")
async def get_mt5_positions() -> Dict[str, Any]:
    """Get current MT5 positions."""
    if not state.mt5_connected:
        return {"success": False, "error": "MT5 not connected"}

    positions = mt5.positions_get()
    if positions is None:
        positions = []

    position_data = []
    for pos in positions:
        position_data.append(
            {
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
                "time": datetime.fromtimestamp(pos.time).isoformat(),
            }
        )

    missing_sl = sum(1 for p in position_data if not p["has_sl"])
    missing_tp = sum(1 for p in position_data if not p["has_tp"])

    return {
        "success": True,
        "positions": position_data,
        "total": len(position_data),
        "missing_sl": missing_sl,
        "missing_tp": missing_tp,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/mt5/positions/fix-sl-tp")
async def fix_positions_sl_tp() -> Dict[str, Any]:
    """Fix positions that are missing SL/TP."""
    if not state.mt5_connected:
        return {"success": False, "error": "MT5 not connected"}

    await _auto_fix_sl_tp()

    return {
        "success": True,
        "message": "SL/TP fix initiated",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/mt5/chart-data/{symbol}")
async def get_mt5_chart_data(
    symbol: str, timeframe: str = "M5", count: int = 50
) -> Dict[str, Any]:
    """Get MT5 chart data."""
    if not state.mt5_connected:
        return {"success": False, "error": "MT5 not connected"}

    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }

    tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)

    if rates is None or len(rates) == 0:
        return {"success": False, "error": "No data available"}

    chart_data = []
    for rate in rates:
        ts_utc = datetime.utcfromtimestamp(int(rate["time"])).replace(tzinfo=timezone.utc)
        ts_local = ts_utc.astimezone()
        chart_data.append(
            {
                "time": ts_local.strftime("%H:%M"),
                "ts": int(ts_utc.timestamp()),
                "open": float(rate["open"]),
                "high": float(rate["high"]),
                "low": float(rate["low"]),
                "close": float(rate["close"]),
                "volume": int(rate["tick_volume"]),
            }
        )

    return {
        "success": True,
        "symbol": symbol,
        "timeframe": timeframe,
        "data": chart_data,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/mt5/deals/recent")
async def get_recent_mt5_deals(limit: int = 10) -> Dict[str, Any]:
    """Get recent MT5 deals."""
    if not state.mt5_connected:
        return {"success": False, "error": "MT5 not connected"}

    from_date = datetime.now() - timedelta(days=1)
    to_date = datetime.now()
    deals = mt5.history_deals_get(from_date, to_date)

    if deals is None:
        deals = []

    deals = sorted(deals, key=lambda x: x.time, reverse=True)[:limit]

    deal_data = []
    for deal in deals:
        deal_data.append(
            {
                "ticket": deal.ticket,
                "symbol": deal.symbol,
                "type": "BUY" if deal.type == mt5.DEAL_TYPE_BUY else "SELL",
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "time": deal.time,
            }
        )

    return {
        "success": True,
        "deals": deal_data,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Risk
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/risk/overview")
async def risk_overview() -> Dict[str, Any]:
    """Get risk overview."""
    perf = state.performance_metrics
    base_metrics = {
        "current_drawdown": perf.get("current_drawdown", 0.0),
        "max_drawdown": perf.get("max_drawdown", 0.0),
        "sharpe_ratio": perf.get("sharpe_ratio", 0.0),
        "win_rate": perf.get("win_rate", 0.0),
    }

    risk_metrics = get_bus_value_with_fallback("risk_metrics", "BackendAPI", default={}) or {}
    risk_level = get_bus_value_with_fallback("risk_level", "BackendAPI", default="UNKNOWN")
    risk_scale = get_bus_value_with_fallback("risk_scale", "BackendAPI", default=1.0)

    risk_controller = state.module_states.get("risk_controller", {})

    return {
        "success": True,
        **base_metrics,
        **risk_metrics,
        "risk_level": risk_level,
        "risk_scale": risk_scale,
        "system_status": risk_controller.get("status", "unknown"),
        "var_95": risk_controller.get("var_95", 0.0),
        "var_99": risk_controller.get("var_99", 0.0),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/risk/alerts")
async def risk_alerts() -> Dict[str, Any]:
    """Get risk-related alerts."""
    alerts = {
        "anomaly_alerts": get_bus_value_with_fallback("anomaly_alerts", "BackendAPI", default=[]) or [],
        "compliance_alerts": get_bus_value_with_fallback("compliance_violations", "BackendAPI", default=[]) or [],
        "risk_alerts": get_bus_value_with_fallback("risk_alerts", "BackendAPI", default=[]) or [],
        "system_alerts": [a for a in state.alerts if a.get("module") == "risk"],
    }

    total_alerts = sum(len(alert_list) for alert_list in alerts.values())
    critical_count = sum(
        1
        for alert_list in alerts.values()
        for alert in alert_list
        if alert.get("severity") == "critical"
    )

    return {
        "success": True,
        "alerts": alerts,
        "summary": {
            "total_alerts": total_alerts,
            "critical_count": critical_count,
        },
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Memory
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/memory/overview")
async def memory_overview() -> Dict[str, Any]:
    """Get memory system overview."""
    unified_metrics = get_bus_value_with_fallback("unified_metrics", "BackendAPI", default={}) or {}
    memory_status = get_bus_value_with_fallback("unified_memory_status", "BackendAPI", default={}) or {}

    overview = {
        "total_memories": unified_metrics.get("total_memories", 0),
        "memory_utilization": unified_metrics.get("memory_utilization", 0.0),
        "components_active": unified_metrics.get("components_active", 0),
        "processing_status": unified_metrics.get("processing_status", "unknown"),
        "health_status": unified_metrics.get("health_status", "unknown"),
        "components_enabled": memory_status.get("components_enabled", 0),
        "status": memory_status.get("status", "unknown"),
    }

    return sanitize_for_json({"success": True, **overview, "timestamp": datetime.now().isoformat()})


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints - Voting
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/api/voting/overview")
async def voting_overview() -> Dict[str, Any]:
    """Get voting system overview."""
    voting_metrics = get_bus_value_with_fallback("voting_metrics", "BackendAPI", default={}) or {}
    consensus_score = get_bus_value_with_fallback("consensus_score", "BackendAPI", default=0.0) or 0.0
    voting_result = get_bus_value_with_fallback("voting_result", "BackendAPI", default={}) or {}

    successful_ticks = voting_metrics.get("successful_ticks", 0)
    total_ticks = voting_metrics.get("total_ticks", 1)
    success_rate = successful_ticks / max(total_ticks, 1)

    if success_rate >= 0.9:
        health_status = "healthy"
    elif success_rate >= 0.7:
        health_status = "warning"
    else:
        health_status = "critical"

    overview = {
        "total_decisions": total_ticks,
        "successful_decisions": successful_ticks,
        "success_rate": success_rate,
        "health_status": health_status,
        "current_consensus": consensus_score,
        "processing_time_ms": voting_metrics.get("avg_processing_time_ms", 0.0),
        "voting_result": voting_result,
    }

    return {"success": True, **overview, "timestamp": datetime.now().isoformat()}


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Endpoint
# ══════════════════════════════════════════════════════════════════════════════


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    state.websocket_connections.append(websocket)
    logger.info(f"[WS] New connection, total: {len(state.websocket_connections)}")

    try:
        await broadcast_system_state()

        while True:
            try:
                data = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                break

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue

            if message.get("type") == "ping":
                await websocket.send_json(
                    {"type": "pong", "timestamp": datetime.now().isoformat()}
                )
            elif message.get("type") == "request_update":
                await broadcast_system_state()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)
            logger.info(f"[WS] Connection removed, total: {len(state.websocket_connections)}")


# ══════════════════════════════════════════════════════════════════════════════
# Health & Info Endpoints
# ══════════════════════════════════════════════════════════════════════════════


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": state.get_uptime(),
        "system_status": state.system_status,
        "mt5_connected": state.mt5_connected,
        "model_loaded": state.model_loaded,
        "active_connections": len(state.websocket_connections),
        "error_count": len(state.errors),
        "session_id": state.current_session_id,
        "version": "4.0.0",
    }


@app.get("/api")
async def api_documentation() -> Dict[str, Any]:
    """API documentation."""
    return {
        "name": "AI Trading Dashboard API",
        "version": "4.0.0",
        "description": "Live MT5 trading system with comprehensive monitoring",
        "features": [
            "Live MT5 trading with PPO policy",
            "Comprehensive module monitoring",
            "Real-time WebSocket updates",
            "Risk management",
            "Emergency stop controls",
            "Performance analytics",
        ],
        "endpoints": {
            "authentication": {
                "POST /api/login": "Login to MT5",
                "POST /api/logout": "Logout and cleanup",
            },
            "trading": {
                "POST /api/trading/start": "Start live trading",
                "POST /api/trading/stop": "Stop live trading",
                "POST /api/trading/emergency-stop": "Emergency stop",
            },
            "monitoring": {
                "GET /api/status": "System status",
                "GET /api/modules": "List modules",
                "GET /api/modules/{name}": "Module details",
                "POST /api/modules/{name}/toggle": "Toggle module",
                "GET /api/performance": "Performance metrics",
                "GET /api/alerts": "System alerts",
            },
            "mt5": {
                "GET /api/mt5/status": "MT5 status",
                "GET /api/mt5/account": "Account info",
                "GET /api/mt5/positions": "Open positions",
                "GET /api/mt5/chart-data/{symbol}": "Chart data",
            },
            "realtime": {
                "WS /ws": "WebSocket updates",
            },
        },
        "documentation": "/docs",
        "session_id": state.current_session_id,
        "timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Static Frontend
# ══════════════════════════════════════════════════════════════════════════════

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info(f"[OK] Frontend served from: {frontend_dist}")
else:
    logger.warning("[WARN] Frontend build not found. Run: cd frontend && npm run build")

    @app.get("/", response_class=HTMLResponse)
    async def serve_fallback() -> str:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Dashboard - Build Required</title>
            <style>
                body { background: #1f2937; color: white; font-family: sans-serif;
                       display: flex; align-items: center; justify-content: center;
                       min-height: 100vh; margin: 0; }
                .container { text-align: center; padding: 2rem; }
                h1 { color: #60a5fa; }
                code { background: #111827; padding: 1rem; border-radius: 0.5rem; display: block; margin: 1rem 0; }
                a { color: #60a5fa; margin: 0 1rem; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>AI Trading Dashboard</h1>
                <p>Backend is running. Frontend needs to be built:</p>
                <code>cd frontend && npm install && npm run build</code>
                <div>
                    <a href="/docs">API Docs</a>
                    <a href="/health">Health Check</a>
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
        log_level="info",
        access_log=True,
    )
