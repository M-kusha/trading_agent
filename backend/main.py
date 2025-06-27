# backend/main.py
"""
Advanced AI Trading System Dashboard Backend
FastAPI server with full module integration and PPO-only implementation
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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import glob

import pandas as pd
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import MetaTrader5 as mt5

# Windows encoding fix
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure logging with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/backend.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("TradingDashboard")

app = FastAPI(title="AI Trading Dashboard", version="3.0.0")

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
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
    server: str

class TrainingConfig(BaseModel):
    """PPO-only training configuration"""
    timesteps: int = 100000
    learning_rate: float = 3e-4
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    n_steps: int = 2048
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.01
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    debug: bool = False

class LiveTradingConfig(BaseModel):
    """Live trading configuration"""
    instruments: List[str] = ["EURUSD", "XAUUSD"]
    timeframes: List[str] = ["H1", "H4", "D1"]
    update_interval: int = 5
    max_position_size: float = 0.1
    max_total_exposure: float = 0.3
    min_trade_interval: int = 60
    use_trailing_stop: bool = True
    debug: bool = False

class SystemState(BaseModel):
    """System state for frontend display"""
    status: str
    mt5_connected: bool
    model_loaded: bool
    active_positions: int
    total_exposure: float
    current_balance: float
    daily_pnl: float
    risk_level: str
    last_update: str

# ═══════════════════════════════════════════════════════════════════
# Global State Management
# ═══════════════════════════════════════════════════════════════════

class TradingSystemState:
    """Centralized state management"""
    def __init__(self):
        # System status
        self.system_status = "IDLE"
        self.mt5_connected = False
        self.model_loaded = False
        
        # Processes
        self.training_process = None
        self.trading_task = None
        self.tensorboard_process = None
        
        # Trading state
        self.live_env = None
        self.model = None
        self.last_trade_time = {}
        self.performance_metrics = {
            "start_balance": 0.0,
            "current_balance": 0.0,
            "peak_balance": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "total_pnl": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
        }
        
        # Module states
        self.module_states = {
            "position_manager": {},
            "risk_controller": {},
            "strategy_arbiter": {},
            "execution_monitor": {},
            "correlation_controller": {},
            "drawdown_rescue": {},
            "theme_detector": {},
            "memory_systems": {},
            "voting_committee": {},
        }
        
        # WebSocket connections
        self.websocket_connections = []
        
        # Error tracking
        self.errors = []
        self.warnings = []
        
    def add_error(self, error: str):
        self.errors.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
        # Keep only last 100 errors
        self.errors = self.errors[-100:]
        
    def add_warning(self, warning: str):
        self.warnings.append({
            "timestamp": datetime.now().isoformat(),
            "warning": warning
        })
        self.warnings = self.warnings[-100:]

state = TradingSystemState()

# ═══════════════════════════════════════════════════════════════════
# MT5 Integration
# ═══════════════════════════════════════════════════════════════════

def connect_mt5(login: int, password: str, server: str) -> Dict[str, Any]:
    """Connect to MetaTrader 5"""
    try:
        if not mt5.initialize():
            return {"success": False, "error": "Failed to initialize MT5"}
        
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            error_code = mt5.last_error()
            mt5.shutdown()
            return {"success": False, "error": f"Login failed: {error_code}"}
        
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            return {"success": False, "error": "Failed to get account info"}
        
        state.mt5_connected = True
        state.performance_metrics["start_balance"] = account_info.balance
        state.performance_metrics["current_balance"] = account_info.balance
        
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
            }
        }
        
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return {"success": False, "error": str(e)}

def disconnect_mt5():
    """Disconnect from MetaTrader 5"""
    if state.mt5_connected:
        mt5.shutdown()
        state.mt5_connected = False

# ═══════════════════════════════════════════════════════════════════
# Live Trading Integration
# ═══════════════════════════════════════════════════════════════════

async def start_live_trading(config: LiveTradingConfig):
    """Start live trading with PPO model"""
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")
        
        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Trading already running")
        
        # Load PPO model
        model_path = "models/ppo_final_model.zip"
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="PPO model not found")
        
        # Import here to avoid circular imports
        from stable_baselines3 import PPO
        from envs.env import EnhancedTradingEnv, TradingConfig
        from live.live_connector import LiveDataConnector
        
        # Create live connector
        connector = LiveDataConnector(
            instruments=config.instruments,
            timeframes=config.timeframes
        )
        connector.connect()
        
        # Get historical data
        hist_data = connector.get_historical_data(n_bars=1000)
        
        # Create environment with live configuration
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
        
        # Start trading loop
        state.trading_task = asyncio.create_task(
            live_trading_loop(config, connector)
        )
        
        state.system_status = "TRADING"
        logger.info("Live trading started with PPO model")
        
        return {"success": True, "message": "Live trading started"}
        
    except Exception as e:
        error_msg = f"Failed to start live trading: {str(e)}"
        state.add_error(error_msg)
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

async def live_trading_loop(config: LiveTradingConfig, connector):
    """Main live trading loop"""
    try:
        obs, _ = state.live_env.reset()
        step_count = 0
        
        while state.system_status == "TRADING":
            # Get latest market data
            new_data = connector.get_historical_data(n_bars=1)
            
            # Update environment with new data
            for inst in config.instruments:
                inst_key = inst[:3] + "/" + inst[3:] if len(inst) == 6 else inst
                for tf in config.timeframes:
                    if inst_key in new_data and tf in new_data[inst_key]:
                        state.live_env.data[inst_key][tf] = pd.concat([
                            state.live_env.data[inst_key][tf].iloc[1:],
                            new_data[inst_key][tf].iloc[-1:]
                        ])
            
            # Get model prediction
            action, _ = state.model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = state.live_env.step(action)
            
            # Update module states from info
            update_module_states(info)
            
            # Broadcast updates
            await broadcast_system_state()
            
            # Check for emergency conditions
            if check_emergency_conditions():
                await emergency_stop()
                break
            
            step_count += 1
            
            # Sleep before next iteration
            await asyncio.sleep(config.update_interval)
            
    except Exception as e:
        error_msg = f"Trading loop error: {str(e)}"
        state.add_error(error_msg)
        logger.error(error_msg)
        state.system_status = "ERROR"
    finally:
        connector.disconnect()

def update_module_states(info: Dict[str, Any]):
    """Extract and update module states from environment info"""
    try:
        # Position Manager state
        if "position_manager" in info:
            state.module_states["position_manager"] = {
                "open_positions": info["position_manager"].get("open_positions", {}),
                "total_exposure": info["position_manager"].get("total_exposure", 0.0),
                "position_count": info["position_manager"].get("position_count", 0),
                "average_holding_time": info["position_manager"].get("avg_holding_time", 0),
            }
        
        # Risk Controller state
        if "risk" in info:
            state.module_states["risk_controller"] = {
                "risk_scale": info["risk"].get("risk_scale", 1.0),
                "volatility": info["risk"].get("volatility", {}),
                "var_95": info["risk"].get("var_95", 0.0),
                "drawdown": info["risk"].get("drawdown", 0.0),
                "risk_level": info["risk"].get("risk_level", "NORMAL"),
            }
        
        # Strategy Arbiter votes
        if "votes" in info:
            state.module_states["strategy_arbiter"] = {
                "consensus": info["votes"].get("consensus", 0.0),
                "member_votes": info["votes"].get("member_votes", {}),
                "weights": info["votes"].get("weights", []),
                "gate_status": info["votes"].get("gate_status", "OPEN"),
            }
        
        # Execution Monitor
        if "execution" in info:
            state.module_states["execution_monitor"] = {
                "slippage": info["execution"].get("slippage", 0.0),
                "fill_rate": info["execution"].get("fill_rate", 1.0),
                "avg_spread": info["execution"].get("avg_spread", 0.0),
                "execution_quality": info["execution"].get("quality_score", 1.0),
            }
        
        # Theme Detector
        if "themes" in info:
            state.module_states["theme_detector"] = {
                "active_themes": info["themes"].get("active", []),
                "theme_strengths": info["themes"].get("strengths", {}),
                "market_regime": info["themes"].get("regime", "NEUTRAL"),
            }
        
        # Memory Systems
        if "memory" in info:
            state.module_states["memory_systems"] = {
                "mistake_count": info["memory"].get("mistakes", 0),
                "playbook_size": info["memory"].get("playbook_size", 0),
                "memory_usage": info["memory"].get("usage_pct", 0.0),
                "compression_ratio": info["memory"].get("compression", 1.0),
            }
        
    except Exception as e:
        logger.error(f"Error updating module states: {e}")

def check_emergency_conditions() -> bool:
    """Check for emergency stop conditions"""
    # Check drawdown
    if state.module_states["risk_controller"].get("drawdown", 0) > 0.3:
        logger.warning("Emergency: Maximum drawdown exceeded")
        return True
    
    # Check correlation risk
    if state.module_states["correlation_controller"].get("max_correlation", 0) > 0.9:
        logger.warning("Emergency: Extreme correlation detected")
        return True
    
    # Check system errors
    if len(state.errors) > 50:
        logger.warning("Emergency: Too many system errors")
        return True
    
    return False

# ═══════════════════════════════════════════════════════════════════
# Training Management (PPO Only)
# ═══════════════════════════════════════════════════════════════════

async def start_training(config: TrainingConfig):
    """Start PPO training process"""
    try:
        if state.training_process and state.training_process.poll() is None:
            raise HTTPException(status_code=400, detail="Training already running")
        
        if state.trading_task and not state.trading_task.done():
            raise HTTPException(status_code=400, detail="Cannot train while trading")
        
        # Build training command for PPO only
        cmd = [
            sys.executable, 
            "train/train_ppo_stable.py",  # New consolidated PPO training script
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
        ]
        
        if config.debug:
            cmd.append("--debug")
        
        # Start training process
        state.training_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        state.system_status = "TRAINING"
        logger.info(f"PPO training started with PID: {state.training_process.pid}")
        
        return {"success": True, "pid": state.training_process.pid}
        
    except Exception as e:
        error_msg = f"Training start error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# ═══════════════════════════════════════════════════════════════════
# WebSocket Management
# ═══════════════════════════════════════════════════════════════════

async def broadcast_system_state():
    """Broadcast current system state to all WebSocket clients"""
    if not state.websocket_connections:
        return
    
    # Collect comprehensive system state
    system_state = {
        "type": "system_state",
        "data": {
            "status": state.system_status,
            "mt5_connected": state.mt5_connected,
            "model_loaded": state.model_loaded,
            "performance": state.performance_metrics,
            "modules": state.module_states,
            "timestamp": datetime.now().isoformat(),
        }
    }
    
    # Send to all connected clients
    disconnected = []
    for websocket in state.websocket_connections:
        try:
            await websocket.send_json(system_state)
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for ws in disconnected:
        state.websocket_connections.remove(ws)

# ═══════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    # Create required directories
    directories = [
        "logs", "logs/training", "logs/risk", "logs/simulation",
        "logs/strategy", "logs/position", "logs/tensorboard",
        "checkpoints", "models", "data"
    ]
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Start background tasks
    asyncio.create_task(metrics_collector())
    
    logger.info("Trading Dashboard Backend Started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Stop trading
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        await state.trading_task
    
    # Disconnect MT5
    disconnect_mt5()
    
    # Terminate processes
    for process in [state.training_process, state.tensorboard_process]:
        if process and process.poll() is None:
            process.terminate()
            
    logger.info("Trading Dashboard Backend Shutdown")

# Background task for periodic metrics collection
async def metrics_collector():
    """Collect and broadcast metrics periodically"""
    while True:
        try:
            if state.system_status in ["TRADING", "TRAINING"]:
                await broadcast_system_state()
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(10)

# Authentication endpoints
@app.post("/api/login")
async def login(request: LoginRequest):
    """Login to MT5"""
    result = connect_mt5(request.login, request.password, request.server)
    if result["success"]:
        await broadcast_system_state()
        return result
    else:
        raise HTTPException(status_code=401, detail=result["error"])

@app.post("/api/logout")
async def logout():
    """Logout and cleanup"""
    # Stop trading if active
    if state.trading_task and not state.trading_task.done():
        state.system_status = "STOPPING"
        await state.trading_task
    
    disconnect_mt5()
    state.system_status = "IDLE"
    await broadcast_system_state()
    return {"success": True}

# Training endpoints (PPO only)
@app.post("/api/training/start")
async def training_start(config: TrainingConfig):
    """Start PPO training"""
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
        await broadcast_system_state()
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="No training process running")

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
        await state.trading_task
        state.system_status = "IDLE"
        await broadcast_system_state()
        return {"success": True}
    else:
        raise HTTPException(status_code=400, detail="No trading process running")

@app.post("/api/trading/emergency-stop")
async def emergency_stop():
    """Emergency stop - close all positions and halt trading"""
    try:
        if state.mt5_connected:
            # Close all positions
            positions = mt5.positions_get()
            if positions:
                for position in positions:
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
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Failed to close position {position.ticket}: {result.comment}")
        
        # Stop trading
        if state.trading_task and not state.trading_task.done():
            state.system_status = "EMERGENCY_STOP"
            await state.trading_task
        
        state.system_status = "STOPPED"
        await broadcast_system_state()
        
        return {"success": True, "message": "Emergency stop executed"}
        
    except Exception as e:
        error_msg = f"Emergency stop error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Monitoring endpoints
@app.get("/api/status")
async def get_status():
    """Get comprehensive system status"""
    return {
        "system_status": state.system_status,
        "mt5_connected": state.mt5_connected,
        "model_loaded": state.model_loaded,
        "performance": state.performance_metrics,
        "modules": state.module_states,
        "errors": state.errors[-10:],  # Last 10 errors
        "warnings": state.warnings[-10:],  # Last 10 warnings
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/modules/{module_name}")
async def get_module_state(module_name: str):
    """Get specific module state"""
    if module_name in state.module_states:
        return {
            "module": module_name,
            "state": state.module_states[module_name],
            "timestamp": datetime.now().isoformat(),
        }
    else:
        raise HTTPException(status_code=404, detail=f"Module {module_name} not found")

@app.get("/api/logs/{category}")
async def get_logs(category: str, lines: int = 100):
    """Get log files for a category"""
    log_dirs = {
        "training": "logs/training",
        "risk": "logs/risk",
        "strategy": "logs/strategy",
        "position": "logs/position",
        "simulation": "logs/simulation",
        "system": "logs",
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
        "timestamp": datetime.now().isoformat(),
    }

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available model checkpoints"""
    checkpoint_dir = "checkpoints"
    checkpoints = []
    
    if os.path.exists(checkpoint_dir):
        for file_path in glob.glob(os.path.join(checkpoint_dir, "*.zip")):
            stat = os.stat(file_path)
            checkpoints.append({
                "name": os.path.basename(file_path),
                "path": file_path,
                "size": stat.st_size,
                "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            })
    
    return {
        "checkpoints": sorted(checkpoints, key=lambda x: x["modified"], reverse=True),
        "timestamp": datetime.now().isoformat(),
    }

@app.post("/api/checkpoints/save")
async def save_checkpoint(name: str = "manual_checkpoint"):
    """Save current model as checkpoint"""
    if not state.model_loaded or state.model is None:
        raise HTTPException(status_code=400, detail="No model loaded")
    
    try:
        checkpoint_path = f"checkpoints/{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        state.model.save(checkpoint_path)
        
        return {
            "success": True,
            "checkpoint": checkpoint_path,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Failed to save checkpoint: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/model/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a new PPO model"""
    try:
        # Validate file extension
        if not file.filename.endswith('.zip'):
            raise HTTPException(status_code=400, detail="Only .zip model files are accepted")
        
        # Save uploaded file
        file_path = f"models/uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        return {
            "success": True,
            "message": "Model uploaded successfully",
            "path": file_path,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        error_msg = f"Failed to upload model: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# TensorBoard endpoint
@app.post("/api/tensorboard/start")
async def start_tensorboard():
    """Start TensorBoard server"""
    try:
        if state.tensorboard_process and state.tensorboard_process.poll() is None:
            return {"success": True, "message": "TensorBoard already running", "url": "http://localhost:6006"}
        
        log_dir = "logs/tensorboard"
        cmd = [sys.executable, "-m", "tensorboard", "--logdir", log_dir, "--port", "6006", "--host", "0.0.0.0"]
        
        state.tensorboard_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it time to start
        await asyncio.sleep(2)
        
        if state.tensorboard_process.poll() is None:
            logger.info("TensorBoard started successfully")
            return {"success": True, "url": "http://localhost:6006"}
        else:
            error = state.tensorboard_process.stderr.read().decode()
            logger.error(f"TensorBoard failed to start: {error}")
            return {"success": False, "error": error}
            
    except Exception as e:
        error_msg = f"Error starting TensorBoard: {str(e)}"
        state.add_error(error_msg)
        return {"success": False, "error": error_msg}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    state.websocket_connections.append(websocket)
    
    try:
        # Send initial state
        await broadcast_system_state()
        
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            # Handle any client messages if needed
            
    except WebSocketDisconnect:
        state.websocket_connections.remove(websocket)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": (datetime.now() - datetime.fromtimestamp(0)).total_seconds(),
    }

# API documentation root - FIXED: This should come BEFORE static file mounting
@app.get("/api")
async def api_root():
    """API documentation"""
    return {
        "name": "AI Trading Dashboard API",
        "version": "3.0.0",
        "endpoints": {
            "auth": {
                "POST /api/login": "Login to MT5",
                "POST /api/logout": "Logout and cleanup",
            },
            "training": {
                "POST /api/training/start": "Start PPO training",
                "POST /api/training/stop": "Stop training",
            },
            "trading": {
                "POST /api/trading/start": "Start live trading",
                "POST /api/trading/stop": "Stop live trading",
                "POST /api/trading/emergency-stop": "Emergency stop all trading",
            },
            "monitoring": {
                "GET /api/status": "Get system status",
                "GET /api/modules/{module_name}": "Get module state",
                "GET /api/logs/{category}": "Get logs",
                "GET /api/checkpoints": "List checkpoints",
                "POST /api/checkpoints/save": "Save checkpoint",
            },
            "tools": {
                "POST /api/tensorboard/start": "Start TensorBoard",
                "POST /api/model/upload": "Upload model file",
            },
            "websocket": {
                "WS /ws": "WebSocket for real-time updates",
            },
        },
        "documentation": "/docs",
    }

# ═══════════════════════════════════════════════════════════════════
# Serve Static Frontend - MUST BE LAST
# ═══════════════════════════════════════════════════════════════════
frontend_dist = Path(__file__).parent.parent / "frontend" / "dist"
if frontend_dist.exists():
    # Mount frontend AFTER all API routes are defined
    app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    logger.info(f"Frontend served from: {frontend_dist}")
else:
    logger.warning("⚠️ Frontend build not found. Run: cd frontend && npm run build")
    
    # Fallback HTML response for root
    from fastapi.responses import HTMLResponse
    
    @app.get("/", response_class=HTMLResponse)
    async def serve_fallback():
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Trading Dashboard - Build Required</title>
            <style>
                body { 
                    background: #111827; 
                    color: white; 
                    font-family: system-ui; 
                    display: flex; 
                    align-items: center; 
                    justify-content: center; 
                    min-height: 100vh;
                    margin: 0;
                }
                .container {
                    text-align: center;
                    padding: 2rem;
                    background: #1f2937;
                    border-radius: 1rem;
                    border: 1px solid #374151;
                }
                code {
                    background: #374151;
                    padding: 0.5rem 1rem;
                    border-radius: 0.5rem;
                    display: inline-block;
                    margin: 0.5rem 0;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Frontend Build Required</h1>
                <p>The frontend has not been built yet.</p>
                <p>To build the frontend, run:</p>
                <code>cd frontend && npm install && npm run build</code>
                <p style="margin-top: 2rem;">
                    <a href="/docs" style="color: #60a5fa;">View API Documentation</a>
                </p>
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
        log_level="info"
    )