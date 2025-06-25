# backend/main.py
"""
Advanced AI Trading System Dashboard Backend
FastAPI server with real-time monitoring, control, and MT5 integration
Windows-compatible version with proper encoding handling
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
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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

app = FastAPI(title="AI Trading Dashboard", version="2.0.0")

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
    model_type: str = "ppo"
    timesteps: int = 100000
    learning_rate: float = 0.0003
    batch_size: int = 64
    debug: bool = True

class TradingConfig(BaseModel):
    symbols: List[str] = ["EURUSD", "XAUUSD"]
    max_risk_per_trade: float = 0.02
    max_drawdown: float = 0.20
    emergency_stop: bool = False

# ═══════════════════════════════════════════════════════════════════
# Global State Management
# ═══════════════════════════════════════════════════════════════════

class SystemState:
    def __init__(self):
        self.mt5_connected = False
        self.mt5_account = None
        self.training_process = None
        self.trading_process = None
        self.tensorboard_process = None
        self.system_status = "IDLE"  # IDLE, TRAINING, TRADING, PAUSED, ERROR
        self.session_data = {}
        self.websocket_connections = []
        self.last_metrics = {}
        self.error_log = []
        
    def add_error(self, error: str):
        self.error_log.append({
            "timestamp": datetime.now().isoformat(),
            "error": error
        })
        if len(self.error_log) > 100:
            self.error_log.pop(0)

state = SystemState()
executor = ThreadPoolExecutor(max_workers=4)

# ═══════════════════════════════════════════════════════════════════
# MT5 Connection Management
# ═══════════════════════════════════════════════════════════════════

def connect_mt5(login: int, password: str, server: str) -> Dict[str, Any]:
    """Initialize MT5 connection"""
    try:
        if not mt5.initialize():
            return {"success": False, "error": "MT5 initialization failed"}
        
        # Attempt login
        authorized = mt5.login(login, password=password, server=server)
        if not authorized:
            error_code = mt5.last_error()
            mt5.shutdown()
            return {"success": False, "error": f"Login failed: {error_code}"}
        
        # Get account info
        account_info = mt5.account_info()
        if account_info is None:
            mt5.shutdown()
            return {"success": False, "error": "Failed to get account info"}
        
        state.mt5_connected = True
        state.mt5_account = {
            "login": account_info.login,
            "balance": account_info.balance,
            "equity": account_info.equity,
            "margin": account_info.margin,
            "free_margin": account_info.margin_free,
            "server": account_info.server,
            "currency": account_info.currency,
            "leverage": account_info.leverage,
        }
        
        logger.info(f"MT5 connected successfully: {account_info.login} on {account_info.server}")
        return {"success": True, "account": state.mt5_account}
        
    except Exception as e:
        logger.error(f"MT5 connection error: {e}")
        return {"success": False, "error": str(e)}

def disconnect_mt5():
    """Disconnect from MT5"""
    try:
        if state.mt5_connected:
            mt5.shutdown()
            state.mt5_connected = False
            state.mt5_account = None
            logger.info("MT5 disconnected")
    except Exception as e:
        logger.error(f"MT5 disconnect error: {e}")

# ═══════════════════════════════════════════════════════════════════
# Process Management
# ═══════════════════════════════════════════════════════════════════

def start_tensorboard():
    """Start TensorBoard server"""
    try:
        if state.tensorboard_process and state.tensorboard_process.poll() is None:
            return {"success": True, "message": "TensorBoard already running"}
        
        log_dir = "logs/tensorboard"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Check if there are any event files
        event_files = glob.glob(os.path.join(log_dir, "**/events.out.*"), recursive=True)
        if not event_files:
            logger.warning("No TensorBoard event files found")
        
        cmd = [sys.executable, "-m", "tensorboard", "--logdir", log_dir, "--port", "6006", "--host", "0.0.0.0"]
        state.tensorboard_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Give it time to start
        time.sleep(2)
        
        if state.tensorboard_process.poll() is None:
            logger.info("TensorBoard started successfully on port 6006")
            return {"success": True, "url": "http://localhost:6006"}
        else:
            error = state.tensorboard_process.stderr.read().decode()
            logger.error(f"TensorBoard failed to start: {error}")
            return {"success": False, "error": error}
            
    except Exception as e:
        logger.error(f"Error starting TensorBoard: {e}")
        return {"success": False, "error": str(e)}

def stop_process(process_name: str):
    """Stop a named process"""
    try:
        if process_name == "training" and state.training_process:
            state.training_process.terminate()
            state.training_process.wait(timeout=10)
            state.training_process = None
            state.system_status = "IDLE"
            
        elif process_name == "trading" and state.trading_process:
            state.trading_process.terminate()
            state.trading_process.wait(timeout=10)
            state.trading_process = None
            state.system_status = "IDLE"
            
        elif process_name == "tensorboard" and state.tensorboard_process:
            state.tensorboard_process.terminate()
            state.tensorboard_process.wait(timeout=5)
            state.tensorboard_process = None
            
        logger.info(f"Stopped {process_name} process")
        return {"success": True}
        
    except Exception as e:
        logger.error(f"Error stopping {process_name}: {e}")
        return {"success": False, "error": str(e)}

# ═══════════════════════════════════════════════════════════════════
# File and Log Management
# ═══════════════════════════════════════════════════════════════════

def get_log_files(pattern: str = "*") -> List[Dict[str, Any]]:
    """Get available log files"""
    log_files = []
    
    # Standard log directories
    log_dirs = [
        "logs",
        "logs/training",
        "logs/risk",
        "logs/simulation", 
        "logs/strategy",
        "logs/position",
        "logs/tensorboard"
    ]
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for file_path in glob.glob(os.path.join(log_dir, f"**/{pattern}.log"), recursive=True):
                stat = os.stat(file_path)
                log_files.append({
                    "path": file_path,
                    "name": os.path.basename(file_path),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "category": os.path.basename(os.path.dirname(file_path))
                })
    
    return sorted(log_files, key=lambda x: x["modified"], reverse=True)

def tail_file(file_path: str, lines: int = 100) -> List[str]:
    """Get last N lines from a file"""
    try:
        if not os.path.exists(file_path):
            return [f"File not found: {file_path}"]
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.readlines()[-lines:]
    except Exception as e:
        return [f"Error reading file: {e}"]

def get_audit_data(component: str, limit: int = 50) -> List[Dict[str, Any]]:
    """Read JSONL audit files"""
    audit_files = {
        "portfolio": "logs/risk/portfolio_risk_audit.jsonl",
        "execution": "logs/risk/execution_quality_audit.jsonl", 
        "opponent": "logs/simulation/opponent_simulator_audit.jsonl",
        "trade_explanation": "logs/auditing/trade_explanation_audit.jsonl",
    }
    
    file_path = audit_files.get(component)
    if not file_path or not os.path.exists(file_path):
        return []
    
    try:
        entries = []
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        
        return entries[-limit:] if entries else []
    except Exception as e:
        logger.error(f"Error reading audit file {file_path}: {e}")
        return []

# ═══════════════════════════════════════════════════════════════════
# Metrics Collection
# ═══════════════════════════════════════════════════════════════════

def collect_system_metrics() -> Dict[str, Any]:
    """Collect comprehensive system metrics"""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "system_status": state.system_status,
        "mt5_connected": state.mt5_connected,
        "processes": {
            "training": state.training_process is not None and state.training_process.poll() is None,
            "trading": state.trading_process is not None and state.trading_process.poll() is None,
            "tensorboard": state.tensorboard_process is not None and state.tensorboard_process.poll() is None,
        }
    }
    
    # MT5 account metrics
    if state.mt5_connected and state.mt5_account:
        try:
            # Refresh account info
            account_info = mt5.account_info()
            if account_info:
                metrics["account"] = {
                    "balance": account_info.balance,
                    "equity": account_info.equity,
                    "margin": account_info.margin,
                    "free_margin": account_info.margin_free,
                    "profit": account_info.profit,
                    "margin_level": account_info.margin_level if account_info.margin > 0 else 0,
                }
                
                # Get open positions
                positions = mt5.positions_get()
                if positions:
                    metrics["positions"] = [
                        {
                            "symbol": pos.symbol,
                            "type": "BUY" if pos.type == 0 else "SELL",
                            "volume": pos.volume,
                            "price_open": pos.price_open,
                            "price_current": pos.price_current,
                            "profit": pos.profit,
                            "time": datetime.fromtimestamp(pos.time).isoformat()
                        }
                        for pos in positions
                    ]
                else:
                    metrics["positions"] = []
                    
        except Exception as e:
            logger.error(f"Error collecting MT5 metrics: {e}")
            metrics["account_error"] = str(e)
    
    # Training metrics from performance logs
    try:
        perf_log = "logs/training/performance.log"
        if os.path.exists(perf_log):
            lines = tail_file(perf_log, 10)
            # Parse latest performance data
            for line in reversed(lines):
                if "Best Profit:" in line and "Best Sharpe:" in line:
                    # Extract profit and sharpe from log line
                    parts = line.split("|")
                    if len(parts) >= 2:
                        metrics["training"] = {
                            "latest_log": line.strip(),
                            "last_update": datetime.now().isoformat()
                        }
                    break
    except Exception as e:
        logger.error(f"Error collecting training metrics: {e}")
    
    state.last_metrics = metrics
    return metrics

# ═══════════════════════════════════════════════════════════════════
# WebSocket Management
# ═══════════════════════════════════════════════════════════════════

async def broadcast_update(data: Dict[str, Any]):
    """Broadcast update to all connected WebSocket clients"""
    if state.websocket_connections:
        disconnected = []
        for websocket in state.websocket_connections:
            try:
                await websocket.send_json(data)
            except:
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for ws in disconnected:
            state.websocket_connections.remove(ws)

# Background task for periodic metrics collection
async def metrics_collector():
    """Background task to collect and broadcast metrics"""
    while True:
        try:
            metrics = collect_system_metrics()
            await broadcast_update({"type": "metrics", "data": metrics})
            await asyncio.sleep(5)  # Update every 5 seconds
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
            await asyncio.sleep(10)

# ═══════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup_event():
    """Initialize background tasks"""
    asyncio.create_task(metrics_collector())
    
    # Start TensorBoard if log files exist
    tb_result = start_tensorboard()
    if tb_result["success"]:
        logger.info("TensorBoard auto-started")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    disconnect_mt5()
    
    for process in [state.training_process, state.trading_process, state.tensorboard_process]:
        if process and process.poll() is None:
            process.terminate()

# Authentication endpoints
@app.post("/api/login")
async def login(request: LoginRequest):
    """Login to MT5 and initialize session"""
    try:
        result = connect_mt5(request.login, request.password, request.server)
        if result["success"]:
            await broadcast_update({"type": "login_success", "data": result})
            return result
        else:
            state.add_error(f"Login failed: {result['error']}")
            raise HTTPException(status_code=401, detail=result["error"])
    except Exception as e:
        error_msg = f"Login error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/logout")
async def logout():
    """Logout and cleanup"""
    disconnect_mt5()
    await broadcast_update({"type": "logout"})
    return {"success": True}

# System control endpoints
@app.post("/api/training/start")
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start training process"""
    try:
        if state.training_process and state.training_process.poll() is None:
            raise HTTPException(status_code=400, detail="Training already running")
        
        if state.trading_process and state.trading_process.poll() is None:
            raise HTTPException(status_code=400, detail="Cannot train while trading")
        
        # Build training command
        cmd = [
            sys.executable, "train/train_ppo_lag.py",
            "--model", config.model_type,
            "--timesteps", str(config.timesteps),
            "--lr", str(config.learning_rate),
            "--batch_size", str(config.batch_size)
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
        
        logger.info(f"Training started with PID: {state.training_process.pid}")
        await broadcast_update({"type": "training_started", "data": {"config": config.dict()}})
        
        return {"success": True, "pid": state.training_process.pid}
        
    except Exception as e:
        error_msg = f"Training start error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/training/stop")
async def stop_training():
    """Stop training process"""
    result = stop_process("training")
    if result["success"]:
        await broadcast_update({"type": "training_stopped"})
    return result

@app.post("/api/trading/start")
async def start_trading(config: TradingConfig):
    """Start live trading"""
    try:
        if not state.mt5_connected:
            raise HTTPException(status_code=400, detail="MT5 not connected")
            
        if state.trading_process and state.trading_process.poll() is None:
            raise HTTPException(status_code=400, detail="Trading already running")
            
        if state.training_process and state.training_process.poll() is None:
            raise HTTPException(status_code=400, detail="Cannot trade while training")
        
        # Build trading command
        cmd = [sys.executable, "run.py", "--model", "sac"]
        
        # Start trading process
        state.trading_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.getcwd()
        )
        
        state.system_status = "TRADING"
        
        logger.info(f"Trading started with PID: {state.trading_process.pid}")
        await broadcast_update({"type": "trading_started", "data": {"config": config.dict()}})
        
        return {"success": True, "pid": state.trading_process.pid}
        
    except Exception as e:
        error_msg = f"Trading start error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop trading process"""
    result = stop_process("trading")
    if result["success"]:
        await broadcast_update({"type": "trading_stopped"})
    return result

@app.post("/api/trading/emergency_stop")
async def emergency_stop():
    """Emergency stop - close all positions and stop trading"""
    try:
        if state.mt5_connected:
            # Close all open positions
            positions = mt5.positions_get()
            if positions:
                for position in positions:
                    # Create close request
                    symbol = position.symbol
                    lot = position.volume
                    position_type = position.type
                    
                    # Opposite order type for closing
                    order_type = mt5.ORDER_TYPE_SELL if position_type == 0 else mt5.ORDER_TYPE_BUY
                    
                    request = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot,
                        "type": order_type,
                        "position": position.ticket,
                        "deviation": 20,
                        "comment": "Emergency stop",
                    }
                    
                    result = mt5.order_send(request)
                    if result.retcode != mt5.TRADE_RETCODE_DONE:
                        logger.error(f"Failed to close position {position.ticket}: {result.comment}")
        
        # Stop trading process
        stop_result = stop_process("trading")
        
        state.system_status = "EMERGENCY_STOPPED"
        await broadcast_update({"type": "emergency_stop"})
        
        return {"success": True, "message": "Emergency stop executed"}
        
    except Exception as e:
        error_msg = f"Emergency stop error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

# Monitoring endpoints
@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return collect_system_metrics()

@app.get("/api/logs/{category}")
async def get_logs(category: str, lines: int = 100):
    """Get log files for a category"""
    log_files = get_log_files(category)
    
    if not log_files:
        return {"files": [], "content": []}
    
    # Get content from the most recent file
    latest_file = log_files[0]["path"]
    content = tail_file(latest_file, lines)
    
    return {
        "files": log_files,
        "latest_file": latest_file,
        "content": content
    }

@app.get("/api/audit/{component}")
async def get_audit(component: str, limit: int = 50):
    """Get audit data for a component"""
    data = get_audit_data(component, limit)
    return {"component": component, "entries": data}

@app.get("/api/votes")
async def get_votes():
    """Get latest expert voting data"""
    # This would parse voting logs from the strategy modules
    try:
        voting_log = "logs/strategy/voting.log"
        if os.path.exists(voting_log):
            lines = tail_file(voting_log, 20)
            # Parse voting data from logs
            votes = []
            for line in lines:
                if "Vote:" in line or "Consensus:" in line:
                    votes.append(line.strip())
            return {"votes": votes}
        return {"votes": []}
    except Exception as e:
        return {"error": str(e), "votes": []}

@app.get("/api/metrics/correlation")
async def get_correlation_metrics():
    """Get correlation risk metrics"""
    try:
        corr_audit = get_audit_data("portfolio", 10)
        if corr_audit:
            latest = corr_audit[-1]
            return {
                "correlation_score": latest.get("max_correlation", 0),
                "risk_warnings": latest.get("warnings", []),
                "timestamp": latest.get("timestamp")
            }
        return {"correlation_score": 0, "risk_warnings": [], "timestamp": None}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/tensorboard")
async def get_tensorboard_url():
    """Get TensorBoard URL"""
    if state.tensorboard_process and state.tensorboard_process.poll() is None:
        return {"url": "http://localhost:6006", "running": True}
    else:
        # Try to start TensorBoard
        result = start_tensorboard()
        if result["success"]:
            return {"url": result["url"], "running": True}
        else:
            return {"url": None, "running": False, "error": result.get("error")}

# Checkpoint management
@app.post("/api/checkpoint/save")
async def save_checkpoint(name: str = "manual"):
    """Save model checkpoint"""
    try:
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"{name}_{timestamp}"
        
        # This would trigger checkpoint saving in the training process
        # For now, we'll just create a placeholder
        checkpoint_path = os.path.join(checkpoint_dir, f"{checkpoint_name}.json")
        
        checkpoint_data = {
            "name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "system_status": state.system_status,
            "metrics": state.last_metrics
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved: {checkpoint_name}")
        return {"success": True, "checkpoint": checkpoint_name, "path": checkpoint_path}
        
    except Exception as e:
        error_msg = f"Checkpoint save error: {str(e)}"
        state.add_error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    try:
        checkpoint_dir = "checkpoints"
        if not os.path.exists(checkpoint_dir):
            return {"checkpoints": []}
        
        checkpoints = []
        for file_path in glob.glob(os.path.join(checkpoint_dir, "*.json")):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    checkpoints.append({
                        "name": data.get("name", os.path.basename(file_path)),
                        "timestamp": data.get("timestamp"),
                        "path": file_path,
                        "size": os.path.getsize(file_path)
                    })
            except:
                continue
        
        return {"checkpoints": sorted(checkpoints, key=lambda x: x["timestamp"], reverse=True)}
        
    except Exception as e:
        return {"error": str(e), "checkpoints": []}

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    state.websocket_connections.append(websocket)
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        state.websocket_connections.remove(websocket)

# Static files for frontend - Create simple fallback if built frontend doesn't exist
if os.path.exists("frontend/dist"):
    app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="static")
else:
    # Serve a simple HTML page if no built frontend exists
    @app.get("/")
    async def serve_dashboard():
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Trading Dashboard</title>
    <script src="https://unpkg.com/react@18/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@18/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
    <script src="https://unpkg.com/lucide-react@0.300.0/dist/umd/lucide-react.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { margin: 0; padding: 0; background: #111827; color: white; font-family: system-ui; }
        .loading { display: flex; justify-content: center; align-items: center; height: 100vh; }
        .spinner { width: 40px; height: 40px; border: 4px solid #374151; border-top: 4px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div id="root">
        <div class="loading">
            <div style="text-align: center;">
                <div class="spinner"></div>
                <p style="margin-top: 20px; color: #9ca3af;">Loading AI Trading Dashboard...</p>
                <p style="color: #6b7280; font-size: 14px;">Backend running on port 8000</p>
            </div>
        </div>
    </div>
    
    <script type="text/babel">
        const { useState, useEffect } = React;
        const { Brain, Activity, TrendingUp, Shield, Play, Settings } = lucide;
        
        const SimpleDashboard = () => {
            const [status, setStatus] = useState('Loading...');
            const [connected, setConnected] = useState(false);
            
            useEffect(() => {
                // Check backend status
                fetch('/api/status')
                    .then(res => res.json())
                    .then(data => {
                        setStatus('Backend Connected');
                        setConnected(true);
                    })
                    .catch(() => {
                        setStatus('Backend Error');
                    });
            }, []);
            
            return React.createElement('div', {
                className: 'min-h-screen bg-gray-900 text-white p-8'
            }, [
                React.createElement('div', {
                    key: 'header',
                    className: 'text-center mb-8'
                }, [
                    React.createElement('div', {
                        key: 'icon',
                        className: 'w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4'
                    }, React.createElement(Brain, { size: 32, color: 'white' })),
                    React.createElement('h1', {
                        key: 'title',
                        className: 'text-3xl font-bold mb-2'
                    }, 'AI Trading Dashboard'),
                    React.createElement('p', {
                        key: 'status',
                        className: `text-lg ${connected ? 'text-green-400' : 'text-yellow-400'}`
                    }, status)
                ]),
                
                React.createElement('div', {
                    key: 'content',
                    className: 'max-w-4xl mx-auto'
                }, [
                    React.createElement('div', {
                        key: 'grid',
                        className: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8'
                    }, [
                        React.createElement('div', {
                            key: 'card1',
                            className: 'bg-gray-800 rounded-lg p-6 border border-gray-700'
                        }, [
                            React.createElement('div', {
                                key: 'header1',
                                className: 'flex items-center mb-4'
                            }, [
                                React.createElement(Activity, { key: 'icon1', className: 'mr-3 text-blue-400', size: 24 }),
                                React.createElement('h3', { key: 'title1', className: 'text-lg font-semibold' }, 'System Status')
                            ]),
                            React.createElement('p', { key: 'text1', className: 'text-gray-300' }, connected ? 'Backend running successfully' : 'Connecting to backend...')
                        ]),
                        
                        React.createElement('div', {
                            key: 'card2',
                            className: 'bg-gray-800 rounded-lg p-6 border border-gray-700'
                        }, [
                            React.createElement('div', {
                                key: 'header2',
                                className: 'flex items-center mb-4'
                            }, [
                                React.createElement(Shield, { key: 'icon2', className: 'mr-3 text-green-400', size: 24 }),
                                React.createElement('h3', { key: 'title2', className: 'text-lg font-semibold' }, 'API Access')
                            ]),
                            React.createElement('a', {
                                key: 'link2',
                                href: '/docs',
                                className: 'text-blue-400 hover:text-blue-300'
                            }, 'View API Documentation')
                        ]),
                        
                        React.createElement('div', {
                            key: 'card3',
                            className: 'bg-gray-800 rounded-lg p-6 border border-gray-700'
                        }, [
                            React.createElement('div', {
                                key: 'header3',
                                className: 'flex items-center mb-4'
                            }, [
                                React.createElement(Settings, { key: 'icon3', className: 'mr-3 text-purple-400', size: 24 }),
                                React.createElement('h3', { key: 'title3', className: 'text-lg font-semibold' }, 'Quick Setup')
                            ]),
                            React.createElement('p', { key: 'text3', className: 'text-gray-300 text-sm' }, 'Run "npm run build" to enable full dashboard')
                        ])
                    ]),
                    
                    React.createElement('div', {
                        key: 'instructions',
                        className: 'bg-gray-800 rounded-lg p-6 border border-gray-700'
                    }, [
                        React.createElement('h3', {
                            key: 'instTitle',
                            className: 'text-xl font-bold mb-4'
                        }, 'Getting Started'),
                        React.createElement('div', {
                            key: 'steps',
                            className: 'space-y-3'
                        }, [
                            React.createElement('div', { key: 'step1', className: 'flex items-start' }, [
                                React.createElement('span', { key: 'num1', className: 'bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3 mt-0.5' }, '1'),
                                React.createElement('p', { key: 'text1' }, 'Backend is running - you can use the API at /api endpoints')
                            ]),
                            React.createElement('div', { key: 'step2', className: 'flex items-start' }, [
                                React.createElement('span', { key: 'num2', className: 'bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3 mt-0.5' }, '2'),
                                React.createElement('p', { key: 'text2' }, 'Visit /docs for complete API documentation')
                            ]),
                            React.createElement('div', { key: 'step3', className: 'flex items-start' }, [
                                React.createElement('span', { key: 'num3', className: 'bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm mr-3 mt-0.5' }, '3'),
                                React.createElement('p', { key: 'text3' }, 'For full dashboard: run "npm install" then "npm run build"')
                            ])
                        ])
                    ])
                ])
            ]);
        };
        
        ReactDOM.render(React.createElement(SimpleDashboard), document.getElementById('root'));
    </script>
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