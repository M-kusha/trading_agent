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
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import glob

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import MetaTrader5 as mt5
import websockets.server
import websockets

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

# ═══════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════

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
    initial_balance: float = Field(default=10000.0, gt=0)
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

# ═══════════════════════════════════════════════════════════════════
# Global State Management
# ═══════════════════════════════════════════════════════════════════

class EnhancedTradingSystemState:
    """Advanced state management with comprehensive module tracking"""
    
    def __init__(self):
        self.startup_time = datetime.now()
        
        # Core system state
        self.system_status = "IDLE"
        self.mt5_connected = False
        self.model_loaded = False
        self.current_session_id = str(uuid.uuid4())
        
        # Process management
        self.training_process = None
        self.training_websocket_server = None
        self.trading_task = None
        self.tensorboard_process = None
        self.monitoring_tasks = []
        
        # Training state
        self.training_mode = None  # "offline" or "online"
        self.training_metrics = {}
        self.training_metrics_history = []
        self.training_start_time = None
        self.training_config = None
        
        # Trading state
        self.live_env = None
        self.model = None
        self.last_trade_time = {}
        self.trading_config = None
        
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
        
        # Comprehensive module states
        self.module_states = {
            # Core trading modules
            "position_manager": {
                "enabled": True,
                "status": "idle",
                "open_positions": {},
                "total_exposure": 0.0,
                "position_count": 0,
                "avg_holding_time": 0.0,
                "position_sizes": {},
                "instrument_exposures": {},
                "last_signal": None,
                "signal_strength": 0.0,
                "confidence_scores": {},
                "decision_rationale": {},
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Risk management
            "risk_controller": {
                "enabled": True,
                "status": "monitoring",
                "risk_scale": 1.0,
                "risk_level": "NORMAL",
                "volatility": {},
                "var_95": 0.0,
                "var_99": 0.0,
                "drawdown": 0.0,
                "volatility_ratio": 1.0,
                "risk_budget_used": 0.0,
                "freeze_counter": 0,
                "emergency_stops": 0,
                "risk_violations": [],
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Strategy committee
            "strategy_arbiter": {
                "enabled": True,
                "status": "voting",
                "consensus": 0.0,
                "member_votes": {},
                "member_weights": [],
                "gate_status": "OPEN",
                "voting_history": [],
                "collusion_score": 0.0,
                "vote_distribution": {},
                "last_decision": None,
                "decision_confidence": 0.0,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Execution monitoring
            "execution_monitor": {
                "enabled": True,
                "status": "monitoring",
                "slippage": 0.0,
                "fill_rate": 1.0,
                "avg_spread": 0.0,
                "execution_quality": 1.0,
                "latency_ms": 0.0,
                "rejections": 0,
                "partial_fills": 0,
                "execution_costs": 0.0,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Market analysis
            "theme_detector": {
                "enabled": True,
                "status": "analyzing",
                "active_themes": [],
                "theme_strengths": {},
                "market_regime": "NEUTRAL",
                "regime_confidence": 0.0,
                "theme_transitions": [],
                "volatility_regime": "NORMAL",
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Correlation and portfolio risk
            "correlation_controller": {
                "enabled": True,
                "status": "monitoring",
                "correlation_matrix": {},
                "max_correlation": 0.0,
                "risk_concentration": {},
                "diversification_ratio": 1.0,
                "correlation_warnings": [],
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Drawdown protection
            "drawdown_rescue": {
                "enabled": True,
                "status": "monitoring",
                "drawdown_level": 0.0,
                "drawdown_velocity": 0.0,
                "rescue_mode": False,
                "recovery_progress": 0.0,
                "max_drawdown_session": 0.0,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Memory systems
            "memory_systems": {
                "enabled": True,
                "status": "learning",
                "mistake_count": 0,
                "playbook_size": 0,
                "memory_usage": 0.0,
                "compression_ratio": 1.0,
                "learning_rate": 0.0,
                "memory_efficiency": 1.0,
                "pattern_matches": 0,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Anomaly detection
            "anomaly_detector": {
                "enabled": True,
                "status": "scanning",
                "anomaly_score": 0.0,
                "anomalies_detected": [],
                "false_positive_rate": 0.0,
                "detection_sensitivity": 0.5,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
            
            # Regime detection
            "regime_detector": {
                "enabled": True,
                "status": "analyzing",
                "current_regime": "NORMAL",
                "regime_probability": {},
                "regime_duration": 0,
                "regime_changes_today": 0,
                "errors": [],
                "last_update": datetime.now().isoformat()
            },
        }
        
        # WebSocket connections
        self.websocket_connections = []
        self.training_websocket_connections = []
        
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
        
    def add_training_metrics(self, metrics: Dict[str, Any]):
        """Add training metrics and maintain history"""
        self.training_metrics = metrics
        self.training_metrics_history.append({
            **metrics,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 1000 entries to prevent memory issues
        if len(self.training_metrics_history) > 1000:
            self.training_metrics_history = self.training_metrics_history[-1000:]
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get comprehensive training progress data"""
        if not self.training_metrics:
            return {}
            
        return {
            "current_metrics": self.training_metrics,
            "history": self.training_metrics_history[-100:],  # Last 100 entries
            "training_active": self.system_status == "TRAINING",
            "mode": self.training_mode,
            "start_time": self.training_start_time.isoformat() if self.training_start_time else None,
            "elapsed_time": (datetime.now() - self.training_start_time).total_seconds() if self.training_start_time else 0,
        }

# Global state instance
state = EnhancedTradingSystemState()

# ═══════════════════════════════════════════════════════════════════
# Training Metrics WebSocket Server
# ═══════════════════════════════════════════════════════════════════

class TrainingMetricsServer:
    """WebSocket server to receive metrics from training process"""
    
    def __init__(self, host='localhost', port=8001):
        self.host = host
        self.port = port
        self.server = None
        self.clients = set()
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections from training process"""
        self.clients.add(websocket)
        logger.info(f"Training metrics client connected from {websocket.remote_address}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    if data.get("type") == "training_metrics":
                        metrics = data.get("data", {})
                        state.add_training_metrics(metrics)
                        
                        # Broadcast to frontend clients
                        await broadcast_training_metrics(metrics)
                        
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received from training process")
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.clients.remove(websocket)
            logger.info("Training metrics client disconnected")
    
    async def start(self):
        """Start the WebSocket server"""
        self.server = await websockets.server.serve(
            self.handler, self.host, self.port
        )
        logger.info(f"Training metrics server started on ws://{self.host}:{self.port}")
        
    async def stop(self):
        """Stop the WebSocket server"""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

# Global training metrics server
training_metrics_server = TrainingMetricsServer()

# ═══════════════════════════════════════════════════════════════════
# MT5 Integration
# ═══════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════
# Live Trading System
# ═══════════════════════════════════════════════════════════════════

async def start_live_trading(config: LiveTradingConfig):
    """Enhanced live trading with comprehensive monitoring"""
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")
        
        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Trading already active")
        
        # Load PPO model
        model_path = "models/ppo_trading_model.zip"
        if not os.path.exists(model_path):
            # Try alternative paths
            alt_paths = ["models/ppo_final_model.zip", "models/best/best_model.zip"]
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    model_path = alt_path
                    break
            else:
                raise HTTPException(status_code=404, detail="PPO model not found")
        
        logger.info(f"Starting live trading system with model: {model_path}")
        
        # Import trading components
        from stable_baselines3 import PPO
        from envs.env import EnhancedTradingEnv, TradingConfig
        from live.live_connector import LiveDataConnector
        
        # Create live data connector
        connector = LiveDataConnector(
            instruments=config.instruments,
            timeframes=config.timeframes
        )
        connector.connect()
        
        # Get historical data
        hist_data = connector.get_historical_data(n_bars=1000)
        if not hist_data:
            raise HTTPException(status_code=500, detail="Failed to retrieve historical data")
        
        # Create trading environment
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
        
        # Start trading loop
        state.trading_task = asyncio.create_task(
            live_trading_loop(config, connector)
        )
        
        state.system_status = "TRADING"
        state.add_alert("Live trading started successfully", "success", "trading")
        logger.info("Live trading started successfully")
        
        return {"success": True, "message": "Live trading started", "session_id": state.current_session_id}
        
    except Exception as e:
        error_msg = f"Failed to start live trading: {str(e)}"
        state.add_error(error_msg, "trading")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def live_trading_loop(config: LiveTradingConfig, connector):
    """Enhanced live trading loop with comprehensive monitoring"""
    try:
        obs, _ = state.live_env.reset()
        step_count = 0
        last_balance_update = time.time()
        last_health_check = time.time()
        
        logger.info("Live trading loop started")
        
        while state.system_status == "TRADING":
            loop_start = time.time()
            
            try:
                # Update market data
                new_data = connector.get_historical_data(n_bars=1)
                if new_data:
                    update_environment_data(new_data, config)
                
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
        connector.disconnect()
        logger.info("Live trading loop ended")

def update_environment_data(new_data: Dict, config: LiveTradingConfig):
    """Update environment with new market data"""
    try:
        for inst in config.instruments:
            inst_key = inst[:3] + "/" + inst[3:] if len(inst) == 6 else inst
            for tf in config.timeframes:
                if inst_key in new_data and tf in new_data[inst_key]:
                    if hasattr(state.live_env, 'data') and inst_key in state.live_env.data:
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

# ═══════════════════════════════════════════════════════════════════
# Training Management Enhanced
# ═══════════════════════════════════════════════════════════════════

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
            error_output = state.training_process.stderr.read()
            state.add_error(f"Training failed with code {return_code}: {error_output}", "training")
            logger.error(f"Training failed: {error_output}")
        
        state.training_mode = None
        state.training_start_time = None
        
        if state.system_status == "TRAINING":
            state.system_status = "IDLE"
        
    except Exception as e:
        state.add_error(f"Training monitoring error: {str(e)}", "training")

# ═══════════════════════════════════════════════════════════════════
# WebSocket Management Enhanced
# ═══════════════════════════════════════════════════════════════════

async def broadcast_system_state():
    """Broadcast comprehensive system state including training metrics"""
    if not state.websocket_connections:
        return
    
    try:
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
                "timestamp": datetime.now().isoformat(),
            }
        }
        
        # Send to all connected clients
        disconnected = []
        for websocket in state.websocket_connections:
            try:
                await websocket.send_json(system_state)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            if ws in state.websocket_connections:
                state.websocket_connections.remove(ws)
        
        # Update connection count
        state.system_metrics["active_connections"] = len(state.websocket_connections)
        
    except Exception as e:
        state.add_error(f"Broadcast error: {str(e)}", "websocket")

async def broadcast_training_metrics(metrics: Dict[str, Any]):
    """Broadcast training metrics to frontend"""
    if not state.websocket_connections:
        return
        
    try:
        message = {
            "type": "training_metrics",
            "data": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for websocket in state.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)
        
        for ws in disconnected:
            if ws in state.websocket_connections:
                state.websocket_connections.remove(ws)
                
    except Exception as e:
        logger.error(f"Training metrics broadcast error: {e}")

# ═══════════════════════════════════════════════════════════════════
# Emergency Controls
# ═══════════════════════════════════════════════════════════════════

async def emergency_stop():
    """Enhanced emergency stop with comprehensive cleanup"""
    try:
        logger.warning("🚨 EMERGENCY STOP INITIATED")
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
                            logger.info(f"✅ Closed position {position.ticket}")
                        else:
                            logger.error(f"❌ Failed to close position {position.ticket}: {result.comment}")
                            
                    except Exception as e:
                        logger.error(f"Error closing position {position.ticket}: {e}")
        
        # Stop trading loop
        if state.trading_task and not state.trading_task.done():
            state.trading_task.cancel()
            try:
                await state.trading_task
            except asyncio.CancelledError:
                pass
        
        state.system_status = "EMERGENCY_STOPPED"
        state.add_alert("Emergency stop completed", "warning", "emergency")
        logger.warning("🛑 Emergency stop completed")
        
        return {"success": True, "message": "Emergency stop executed"}
        
    except Exception as e:
        error_msg = f"Emergency stop error: {str(e)}"
        state.add_error(error_msg, "emergency")
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ═══════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Enhanced system startup"""
    # Create required directories
    directories = [
        "logs", "logs/training", "logs/risk", "logs/simulation",
        "logs/strategy", "logs/position", "logs/tensorboard",
        "logs/evaluation", "logs/monitoring",
        "checkpoints", "models", "models/best", "data", "data/processed",
        "metrics"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Start training metrics WebSocket server
    asyncio.create_task(training_metrics_server.start())
    
    # Start background monitoring tasks
    state.monitoring_tasks.extend([
        asyncio.create_task(periodic_metrics_collector()),
        asyncio.create_task(system_health_monitor()),
        asyncio.create_task(performance_tracker()),
    ])
    
    logger.info("🚀 Enhanced Trading Dashboard Backend Started")
    state.add_alert("System started successfully", "success", "system")

@app.on_event("shutdown")
async def shutdown_event():
    """Enhanced system shutdown"""
    logger.info("🛑 Shutting down trading dashboard...")
    
    # Stop training metrics server
    await training_metrics_server.stop()
    
    # Cancel monitoring tasks
    for task in state.monitoring_tasks:
        if not task.done():
            task.cancel()
    
    # Stop training if active
    if state.training_process and state.training_process.poll() is None:
        state.training_process.terminate()
        state.training_process.wait(timeout=10)
    
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
    
    logger.info("✅ Trading Dashboard Backend Shutdown Complete")

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

# Training endpoints
@app.post("/api/training/start")
async def training_start(config: PPOTrainingConfig):
    """Start PPO training with mode selection"""
    result = await start_training(config)
    await broadcast_system_state()
    return result

@app.post("/api/training/stop")
async def training_stop():
    """Stop training"""
    if state.training_process and state.training_process.poll() is None:
        state.training_process.terminate()
        state.training_process.wait(timeout=10)
        state.training_process = None
        state.system_status = "IDLE"
        state.add_alert("Training stopped by user", "info", "training")
        await broadcast_system_state()
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="No training process running")

@app.get("/api/training/progress")
async def get_training_progress():
    """Get detailed training progress and metrics"""
    return state.get_training_progress()

@app.get("/api/training/metrics/history")
async def get_training_metrics_history(limit: int = Query(default=100, le=1000)):
    """Get historical training metrics"""
    history = state.training_metrics_history[-limit:]
    
    # Calculate additional statistics
    if history:
        rewards = [m.get('episode_reward_mean', 0) for m in history]
        
        return {
            "metrics": history,
            "statistics": {
                "avg_reward": np.mean(rewards) if rewards else 0,
                "max_reward": max(rewards) if rewards else 0,
                "min_reward": min(rewards) if rewards else 0,
                "reward_trend": "improving" if len(rewards) > 1 and rewards[-1] > rewards[0] else "stable",
                "total_episodes": history[-1].get('episodes', 0) if history else 0,
                "total_timesteps": history[-1].get('timestep', 0) if history else 0,
            },
            "count": len(history),
            "limit": limit
        }
    
    return {"metrics": [], "statistics": {}, "count": 0, "limit": limit}

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
        "training": state.get_training_progress(),
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/modules")
async def list_modules():
    """List all modules with detailed status"""
    return {
        "modules": [
            {
                "name": name,
                "enabled": module.get("enabled", False),
                "status": module.get("status", "unknown"),
                "last_update": module.get("last_update", "never"),
                "metrics": {k: v for k, v in module.items() 
                          if k not in ["enabled", "status", "last_update", "errors"]},
                "error_count": len(module.get("errors", [])),
            }
            for name, module in state.module_states.items()
        ],
        "total_modules": len(state.module_states),
        "active_modules": sum(1 for m in state.module_states.values() if m.get("enabled", False)),
        "timestamp": datetime.now().isoformat(),
    }

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
    if module_name not in state.module_states:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")
    
    current_state = state.module_states[module_name].get("enabled", False)
    state.module_states[module_name]["enabled"] = not current_state
    
    action = "enabled" if not current_state else "disabled"
    state.add_alert(f"Module {module_name} {action}", "info", module_name)
    
    await broadcast_system_state()
    return {
        "module": module_name,
        "enabled": not current_state,
        "message": f"Module {action} successfully"
    }

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
        if not file.filename.endswith('.zip'):
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
            if not file.filename.endswith('.csv'):
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
    
    try:
        # Send initial state
        await broadcast_system_state()
        
        # Keep connection alive and handle client messages
        while True:
            try:
                data = await websocket.receive_text()
                # Handle client messages if needed
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
                elif message.get("type") == "request_update":
                    await broadcast_system_state()
                    
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if websocket in state.websocket_connections:
            state.websocket_connections.remove(websocket)

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
            "training": {
                "POST /api/training/start": "Start PPO training (offline/online)",
                "POST /api/training/stop": "Stop training",
                "GET /api/training/progress": "Get training progress",
                "GET /api/training/metrics/history": "Get training metrics history",
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

# ═══════════════════════════════════════════════════════════════════
# Serve Static Frontend - MUST BE LAST
# ═══════════════════════════════════════════════════════════════════

frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"

if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info(f"✅ Frontend served from: {frontend_dist}")
else:
    logger.warning("⚠️ Frontend build not found. Run: cd frontend && npm run build")
    
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
                <h1>🤖 AI Trading Dashboard</h1>
                <div class="status">
                    ✅ Backend is running successfully!
                </div>
                <p>The frontend needs to be built to access the full dashboard.</p>
                <p>To build the frontend, run:</p>
                <code>cd frontend && npm install && npm run build</code>
                <div class="links">
                    <a href="/docs">📚 API Documentation</a>
                    <a href="/health">💚 Health Check</a>
                    <a href="/api">🔧 API Info</a>
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
        access_log=True
    )