# PropFirm Trading System - Project Mapping Documentation

## Project Overview
This project implements a comprehensive PropFirm trading system with Reinforcement Learning (PPO), Curriculum Learning, and a real-time monitoring dashboard. The system is designed to train trading agents that can pass proprietary trading firm evaluations.

## Directory Structure

### Core Directories

1. **`dashboard/`** - Real-time monitoring dashboard
2. **`envs/`** - Trading environments and curriculum system
3. **`train/`** - Training scripts and controllers
4. **`config/`** - Configuration files
5. **`data/`** - Market data storage
6. **`modules/`** - Shared modules
7. **`utils/`** - Utility functions
8. **`live/`** - Live trading components
9. **`backend/`** - Backend services

## Dashboard System

### `dashboard/server.py`
**Purpose**: FastAPI server for real-time training monitoring and metrics visualization.

**Key Functions**:
- `DashboardConfig` - Configuration for dashboard server
- `MetricThresholds` - Defines thresholds for various metrics (win rate, drawdown, etc.)
- `MetricsReader` - Reads and processes training metrics from JSON files
- `get_requirements_by_stage()` - Fetches curriculum stage requirements
- `start_dashboard_server()` - Starts the dashboard server

**Features**:
- Real-time WebSocket updates for metrics
- REST API for metrics, health checks, curriculum data
- Alert logging and history tracking
- Support for multiple client connections

**Files Used**:
- `logs/training/live_metrics.json` - Main metrics source
- `logs/audit/alerts.jsonl` - Alert logs
- `envs/curriculum/` - For stage requirements

## Training System

### `train/train_prop_firm.py`
**Purpose**: Main training script for PropFirm PPO training with curriculum learning.

**Key Functions**:
- `configure_logging()` - Sets up logging configuration
- `load_market_data()` - Loads and processes market data
- `create_vec_envs()` - Creates vectorized trading environments
- `train_prop_firm_agent()` - Standard PPO training
- `train_curriculum_agent()` - Curriculum-based training
- `run_optuna_optimization()` - Hyperparameter optimization

**Training Modes**:
1. **Standard Training**: Fixed environment configuration
2. **Curriculum Training**: Progressive difficulty stages
3. **Optuna Optimization**: Hyperparameter search with walk-forward validation

**Dependencies**:
- `stable_baselines3` - PPO implementation
- `sb3_contrib` - MaskablePPO for action masking
- `optuna` - Hyperparameter optimization

### `train/controllers/`
**Purpose**: Advanced controllers for training management.

**Files**:
- `PIDController` - PID control for learning rate/entropy
- `SmartEntropyController` - Adaptive entropy control
- `TrainingHealthWatchdog` - Monitors training health

### `train/callbacks/`
**Purpose**: Custom callbacks for training.

**Files**:
- `VecEpisodeTradingCallback` - Episode-based metrics logging
- `CurriculumCheckpointCallback` - Curriculum-aware checkpoints
- `CurriculumTrainingCallback` - Manages curriculum transitions

## Environment System

### `envs/prop_firm_env.py`
**Purpose**: Main trading environment implementing OpenAI Gym interface.

**Key Features**:
- Discrete action space with action masking
- Domain randomization for robustness
- Reward shaping for trading behavior
- Integration with curriculum system

### `envs/curriculum/` - Curriculum Learning System

#### `curriculum_manager.py`
**Purpose**: Manages curriculum progression across stages.

**Key Classes**:
- `CurriculumManager` - Main manager class
- `CurriculumStage` - Enum of stages (EXPLORER to LIVE_READY)
- `CurriculumStageConfig` - Stage-specific configuration

**Key Functions**:
- `on_episode_end()` - Processes episode results
- `check_promotion_criteria()` - Evaluates promotion readiness
- `check_demotion_criteria()` - Evaluates demotion conditions
- `get_effective_stage_config()` - Gets blended config during transitions

**Stages**:
1. EXPLORER - Basic exploration
2. EXPERIMENTER - Simple strategies
3. TREND_STUDENT - Trend following
4. SESSION_STUDENT - Session awareness
5. TIMING_STUDENT - Entry timing
6. INTEGRATOR - Multi-skill integration
7. RISK_MANAGER - Risk management
8. STRATEGIST - Advanced strategies
9. PROFESSIONAL - Professional level
10. LIVE_READY - Ready for live trading

#### `curriculum_config.py`
**Purpose**: Defines curriculum stage configurations and requirements.

**Key Data Classes**:
- `CompetenceThresholds` - Performance thresholds per stage
- `SkillRequirements` - Required trading skills
- `CompositeScoringConfig` - Composite scoring settings
- `AdaptiveThresholdConfig` - Adaptive threshold adjustments

#### Other Curriculum Files:
- `metrics.py` - Metrics tracking and calculations
- `skills.py` - Skill assessment and decomposition
- `protocols.py` - Recovery and review protocols
- `validation_gates.py` - Validation and stress testing
- `regime_skill_assessment.py` - Market regime analysis
- `curriculum_invariants.py` - Consistency checking

## Configuration System

### `config/` Directory

**Key Configuration Files**:

1. **`base.yaml`** - Base configuration for all components
2. **`training.yaml`** - Training-specific settings
3. **`live.yaml`** - Live trading configuration
4. **`risk_policy.yaml`** - Risk management policies
5. **`timing_policy.yaml`** - Timing and session policies
6. **`system_config.yaml`** - System-level settings
7. **`presets.yaml`** - Preset configurations
8. **`explainability_standards.yaml`** - Explainability requirements

**Configuration Structure**:
- Environment parameters (spread, slippage, limits)
- Reward function parameters
- Curriculum stage definitions
- Trading constraints and rules
- Risk management settings

## Data Pipeline

### Data Loading (`train/train_prop_firm.py`)
**Functions**:
- `load_market_data()` - Loads CSV data with timeframe detection
- `_create_synthetic_data()` - Generates synthetic data if no files found
- `build_walk_forward_folds()` - Creates walk-forward validation splits

**Supported Timeframes**: M1, M5, M15, M30, H1, H2, H4, H8, D1, W1

**Data Requirements**:
- OHLCV columns (open, high, low, close, volume)
- Minimum bars per timeframe (e.g., 5000 for M15)

## Live Trading System

### `live/` Directory
**Purpose**: Components for live trading deployment.

**Key Components**:
- Live trading environment
- Risk monitoring
- Order execution
- Performance tracking

## Utility Modules

### `utils/` Directory
**Purpose**: Shared utility functions.

**Key Areas**:
- Data processing helpers
- Mathematical utilities
- Logging utilities
- File I/O operations

### `modules/` Directory
**Purpose**: Shared business logic modules.

**Key Modules**:
- Market analysis tools
- Risk calculators
- Performance metrics
- Reporting utilities

## Execution Pipeline

### Starting Points:

1. **Training Pipeline**: `start_live_training_pipeline.py`
2. **Live Trading**: `start_live_trading.py`
3. **Audit/Explainability**: `explain_audit.py`
4. **CSV Expert Audit**: `start_csv_expert_audit_pipeline.py`

## File Dependencies Map

```
dashboard/server.py
    ├── envs/curriculum/curriculum_config.py (for stage requirements)
    ├── logs/training/live_metrics.json (metrics source)
    └── logs/audit/alerts.jsonl (alert logging)

train/train_prop_firm.py
    ├── envs/prop_firm_env.py (trading environment)
    ├── envs/curriculum/curriculum_manager.py (curriculum system)
    ├── train/controllers/*.py (training controllers)
    ├── train/callbacks/*.py (training callbacks)
    ├── config/*.yaml (configurations)
    └── data/processed/ (market data)

envs/curriculum/curriculum_manager.py
    ├── curriculum_config.py (stage definitions)
    ├── metrics.py (metrics calculations)
    ├── skills.py (skill assessment)
    ├── protocols.py (recovery protocols)
    └── validation_gates.py (validation testing)
```

## Key Integration Points

1. **Dashboard ↔ Training**: Real-time metrics via `live_metrics.json`
2. **Training ↔ Curriculum**: Stage progression via `CurriculumManager`
3. **Environment ↔ Curriculum**: Dynamic difficulty via `get_effective_stage_config()`
4. **Training ↔ Configuration**: YAML configs for hyperparameters
5. **Live Trading ↔ Training**: Model checkpoints and configurations

## Monitoring and Logging

**Log Directories**:
- `logs/training/` - Training metrics and checkpoints
- `logs/audit/` - Audit trails and alerts
- `logs/optuna/` - Hyperparameter optimization results
- `runs/` - TensorBoard logs

**Dashboard Views**:
1. **Overview** - Key metrics and progress
2. **Learning** - Training statistics (losses, entropy, etc.)
3. **Trading** - Trading performance (win rate, PnL, drawdown)
4. **Quality** - Trade quality metrics (R-multiple, profit factor)
5. **Curriculum** - Stage progression and requirements
6. **Alerts** - System alerts and warnings

## Deployment Notes

**Requirements**: See `requirements.txt`

**Starting the Dashboard**:
```bash
python dashboard/server.py --host 0.0.0.0 --port 8765
```

**Starting Training**:
```bash
python train/train_prop_firm.py --curriculum --timesteps 10000000
```

**Starting Curriculum Training**:
```bash
python train/train_prop_firm.py --curriculum --start-stage EXPLORER --goal-based
```

## Troubleshooting

### Common Issues:
1. **Missing Data**: Ensure data files in `data/processed/` with correct format
2. **Dashboard Not Loading**: Check port 8765 is available
3. **Curriculum Not Progressing**: Check threshold requirements in config
4. **Training Instability**: Adjust reward scaling or entropy coefficient

### Debugging Tools:
- Dashboard alert system
- Training logs in `logs/training/train.log`
- Curriculum progress reports via API endpoints
- TensorBoard for training visualization

---
*Last Updated: Based on code exploration of the PropFirm trading system*
*Version: System v2.2 with Curriculum Learning v2.x*