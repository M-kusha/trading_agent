# 🤖 AI Trading System - PPO-Lagrangian Dashboard

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![PPO](https://img.shields.io/badge/RL-PPO--Only-green)](https://stable-baselines3.readthedocs.io)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-red)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/Frontend-React-blue)](https://reactjs.org)
[![MetaTrader5](https://img.shields.io/badge/Broker-MT5-orange)](https://www.metatrader5.com)

> **Production-ready AI trading system with PPO-only implementation, comprehensive module integration, and full frontend/backend dashboard control.**

## 🏗️ **System Architecture**

### **Clean PPO-Only Design**
- ✅ **PPO-Lagrangian** reinforcement learning agent
- 🎯 **Focus:** Single, optimized algorithm with proven performance
- 🔧 **Backend:** FastAPI with real-time WebSocket updates
- 🖥️ **Frontend:** React dashboard with comprehensive module monitoring

### **Core Components**
```
📁 Project Structure (Cleaned)
├── 🚀 run_dashboard.py          # Main Runner
├── 🧹 cleanup_system.py         # Cleanup SAC/TD3 files
├── 📋 requirements.txt          # PPO-only dependencies
│
├── 🖥️ frontend/                 # React dashboard
│   ├── src/App.jsx              # Enhanced UI with module integration
│   ├── src/index.css            # Professional styling
│   └── dist/                    # Production build
│
├── ⚙️ backend/                  # FastAPI server
│   └── main.py                  # Enhanced backend with full module API
│
├── 🧠 train/                    # PPO-only training
│   └── train_ppo_stable.py      # Consolidated PPO training
│
├── 🏢 envs/                     # Trading environment
│   ├── env.py                   # Enhanced environment
│   └── config.py                # PPO-only configuration
│
├── 🔧 modules/                  # Core trading modules
│   ├── position/                # Position management
│   ├── risk/                    # Risk controllers
│   ├── strategy/                # Voting committee
│   ├── memory/                  # Learning systems
│   └── execution/               # Trade execution
│
└── 📊 logs/                     # Organized logging
    ├── training/
    ├── trading/
    ├── risk/
    └── monitoring/
```

## 🚀 **Quick Start**

### **1. System Setup**
```bash
# Clone and setup
git clone <repository>
cd ai-trading-system

# Install dependencies (PPO-only)
pip install -r requirements.txt

# Clean up old SAC/TD3 files (optional)
python cleanup_system.py --scan      # See what would be removed
python cleanup_system.py --live      # Perform cleanup
```

### **2. Start Dashboard**
```bash
# Production mode (recommended)
python run_dashboard.py

# Development mode (separate frontend server)  
python run_dashboard.py --dev

# Debug mode with verbose logging
python run_dashboard.py --debug

# Custom ports
python run_dashboard.py --port 8080 --frontend-port 3001
```

### **3. Access Dashboard**
- 🖥️ **Dashboard:** http://localhost:8000
- 📚 **API Docs:** http://localhost:8000/docs  
- ❤️ **Health:** http://localhost:8000/health

## 🎯 **Complete Workflow**

### **Training Workflow**
```bash
# 1. Start dashboard
python run_dashboard.py

# 2. Login to MT5 in dashboard UI
# 3. Configure PPO parameters in Training tab
# 4. Click "Start Training"
# 5. Monitor progress in real-time
# 6. Save checkpoints as needed
```

### **Live Trading Workflow** 
```bash
# 1. Complete training (above)
# 2. Configure trading parameters in Trading tab
# 3. Click "Start Trading" 
# 4. Monitor positions and performance
# 5. Use emergency stop if needed
```

## 📊 **Dashboard Features**

### **Overview Tab**
- 💰 Real-time P&L and balance
- 📈 Performance charts and metrics
- 🛡️ Risk management status
- ⚡ System health monitoring
- 🚨 Alert notifications

### **Modules Tab**
- 🎯 Position Manager status
- 🛡️ Risk Controller metrics  
- 🗳️ Strategy Committee voting
- 🧠 Memory system analytics
- 📊 Execution quality monitoring
- 🔍 Anomaly detection alerts

### **Training Tab**
- ⚙️ PPO hyperparameter configuration
- 📊 TensorBoard integration
- 💾 Checkpoint management
- 📈 Training progress monitoring
- 🔄 Model upload/download

### **Trading Tab**
- 🎛️ Live trading configuration
- 📊 Position monitoring
- ⚡ Execution quality metrics
- 🚨 Emergency controls
- 📈 Real-time performance

### **Logs Tab**
- 📝 Real-time log streaming
- 🔍 Category-based filtering
- 📊 System, training, risk logs
- 🔄 Auto-refresh capabilities

## 🧠 **PPO Training Guide**

### **Quick Training**
```bash
# Direct training (bypass dashboard)
python train/train_ppo_stable.py --timesteps 100000

# With custom parameters
python train/train_ppo_stable.py \
    --timesteps 500000 \
    --lr 3e-4 \
    --batch_size 64 \
    --n_epochs 10 \
    --debug
```

### **Training Parameters**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `timesteps` | 100000 | Total training steps |
| `learning_rate` | 3e-4 | PPO learning rate |
| `batch_size` | 64 | Training batch size |
| `n_epochs` | 10 | PPO epochs per update |
| `gamma` | 0.99 | Discount factor |
| `clip_range` | 0.2 | PPO clip range |
| `n_steps` | 2048 | Steps per rollout |

### **Advanced Configuration**
```python
# Custom training config
config = {
    "timesteps": 1000000,
    "learning_rate": 2e-4,
    "batch_size": 128,
    "n_epochs": 15,
    "gamma": 0.995,
    "clip_range": 0.15,
    "ent_coef": 0.005,
    "target_kl": 0.015,
}
```

## 🏢 **Module Integration**

### **Complete Module Ecosystem**
The system integrates **40+ specialized modules** with real-time monitoring:

#### **🎯 Position Management**
- Smart position sizing and lifecycle management
- Multi-instrument exposure tracking
- Confidence-based decision making
- Real-time P&L monitoring

#### **🛡️ Risk Management**
- Dynamic risk scaling based on market conditions
- VaR calculations and drawdown protection
- Portfolio correlation monitoring
- Emergency stop mechanisms

#### **🗳️ Strategy Committee (8-Member Voting)**
- **Liquidity Expert:** Market liquidity analysis
- **Position Manager:** Core trading decisions
- **Theme Expert:** Market regime detection
- **Seasonality Expert:** Time-based adjustments
- **Meta-RL Expert:** Advanced learning decisions
- **Monitor Veto:** Risk-based position vetoing
- **Regime Expert:** Market cycle analysis
- **Risk Controller:** Risk-adjusted voting

#### **🧠 Memory & Learning**
- Mistake memory for avoiding repeated errors
- Playbook memory for successful patterns
- Memory compression for efficiency
- Historical replay analysis

#### **📊 Market Analysis**
- Multi-timeframe regime detection
- Theme strength analysis
- Anomaly detection
- Liquidity mapping

### **Real-Time Module Data**
Every module exposes comprehensive metrics accessible via:
- 🖥️ **Dashboard UI:** Visual monitoring and controls
- 🔌 **WebSocket:** Real-time data streaming  
- 📊 **REST API:** Programmatic access
- 📝 **Logs:** Detailed historical records

## 🔧 **API Integration**

### **Core Endpoints**
```python
# Authentication
POST /api/login              # MT5 login
POST /api/logout             # Logout and cleanup

# Training Control  
POST /api/training/start     # Start PPO training
POST /api/training/stop      # Stop training

# Trading Control
POST /api/trading/start      # Start live trading
POST /api/trading/stop       # Stop trading
POST /api/trading/emergency-stop  # Emergency stop

# Monitoring
GET  /api/status             # System status
GET  /api/modules            # All modules status
GET  /api/modules/{name}     # Specific module
POST /api/modules/{name}/toggle  # Enable/disable module
GET  /api/performance        # Performance metrics
GET  /api/alerts             # System alerts
GET  /api/logs/{category}    # Log files

# Model Management
GET  /api/checkpoints        # List checkpoints  
POST /api/checkpoints/save   # Save checkpoint
POST /api/model/upload       # Upload model

# Tools
POST /api/tensorboard/start  # Start TensorBoard
```

### **WebSocket Integration**
```javascript
// Real-time system state
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'system_state') {
        // Update UI with real-time data
        updateDashboard(data.data);
    }
};
```

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# Optional environment configuration
export TRADING_ENV=production        # production/development
export BACKEND_PORT=8000            # Backend port
export FRONTEND_PORT=3000           # Frontend port (dev mode)
export LOG_LEVEL=INFO               # DEBUG/INFO/WARNING/ERROR
export MAX_POSITIONS=10             # Maximum open positions
export EMERGENCY_DRAWDOWN=0.25      # Emergency stop drawdown
```

### **MT5 Configuration**
```python
# MT5 connection settings
MT5_CONFIG = {
    "login": 12345678,              # Your MT5 login
    "password": "your_password",     # Your MT5 password  
    "server": "MetaQuotes-Demo",     # Your MT5 server
    "timeout": 60000,               # Connection timeout
}
```

## 📊 **Performance Monitoring**

### **Key Metrics Tracked**
- 💰 **Financial:** P&L, drawdown, Sharpe ratio, win rate
- ⚡ **Execution:** Fill rates, slippage, spread analysis
- 🛡️ **Risk:** VaR, correlation, exposure limits
- 🧠 **Learning:** Model performance, training metrics
- 🏢 **System:** CPU, memory, disk usage, errors

### **Real-Time Alerts**
- 🚨 **Critical:** Emergency conditions requiring immediate action
- ⚠️ **Warning:** Conditions requiring attention
- ✅ **Success:** Successful operations and milestones
- ℹ️ **Info:** General system information

## 🚀 **Production Deployment**

### **Docker Deployment**
```bash
# Build production image
docker build -t ai-trading-dashboard .

# Run production container
docker run -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    ai-trading-dashboard
```

### **Production Checklist**
- ✅ MT5 connection configured and tested
- ✅ All dependencies installed and verified
- ✅ Frontend built for production (`npm run build`)
- ✅ Environment variables configured
- ✅ Log directories created with proper permissions
- ✅ Backup strategy implemented
- ✅ Monitoring and alerting configured
- ✅ Emergency procedures documented

## 🔍 **Troubleshooting**

### **Common Issues**

#### **🔌 Connection Issues**
```bash
# Check MT5 connection
python -c "import MetaTrader5 as mt5; print('MT5 Available:', mt5.initialize())"

# Check port availability
python -c "import socket; s=socket.socket(); s.bind(('',8000)); print('Port 8000 available')"
```

#### **📦 Dependency Issues**
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check specific packages
python -c "import torch, stable_baselines3, fastapi; print('All packages OK')"
```

#### **🖥️ Frontend Issues**
```bash
# Rebuild frontend
cd frontend
npm install
npm run build
cd ..

# Check build
ls -la frontend/dist/
```

#### **📝 Log Analysis**
```bash
# Check recent errors
tail -50 logs/backend.log | grep ERROR

# Monitor real-time logs  
tail -f logs/backend.log

# Check specific module logs
ls logs/*/
```

## 🤝 **Support & Development**

### **Development Setup**
```bash
# Development dependencies
pip install pytest pytest-asyncio black flake8 mypy

# Run tests
pytest tests/

# Code formatting  
black .

# Type checking
mypy backend/ train/
```

### **Contributing**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

### **Architecture Decisions**
- **PPO-Only:** Simplified from multi-agent to single, optimized algorithm
- **Dashboard Control:** Replaced CLI with web-based interface
- **Module Integration:** All trading modules exposed via API
- **Real-time Updates:** WebSocket integration for live monitoring
- **Production Ready:** Comprehensive error handling and monitoring

## 📜 **License & Disclaimer**

### **License**
This project is licensed under the MIT License - see the LICENSE file for details.

### **Trading Disclaimer**
⚠️ **IMPORTANT:** This is educational software. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always test thoroughly on demo accounts before live trading.

### **Version Information**
- **Version:** 3.0.0 (PPO-Only)
- **Last Updated:** 2025-01-28
- **Python:** 3.9+ required
- **Dependencies:** See requirements.txt

---

## 🎯 **Next Steps**

1. **Setup:** Follow the Quick Start guide above
2. **Training:** Train your PPO model using the dashboard
3. **Testing:** Validate on demo account extensively
4. **Production:** Deploy with proper monitoring and safeguards
5. **Optimization:** Fine-tune based on performance metrics

**Happy Trading! 🚀📈**