# AI Trading System - Copilot Instructions

## Architecture Overview

**PPO-Lagrangian RL trading system** with event-driven **SmartInfoBus** architecture. Trades EUR/USD and XAU/USD across M15/H1/H4/D1 timeframes.

### M15-Primary Architecture
```
M15 = Signal Generation (100% of direction decision)
H1  = Context filter (confidence modifier ONLY)
H4  = Context filter (confidence modifier ONLY)
D1  = Context filter (confidence modifier ONLY)
```

**Key Rule**: M15 generates the direction (long/short/flat). Context timeframes (H1/H4/D1) ONLY modify confidence:
- If context TFs agree with M15 → +15% confidence bonus
- If context TFs disagree with M15 → -20% confidence penalty
- Context TFs **NEVER** override M15 direction

This is because ExitManager closes trades early with tight TP, so H1/H4/D1 trends rarely have time to play out.

### Core Data Flow
```
MarketDataProvider -> FeatureEngines -> VotingExperts -> CommitteeCoordinator 
    -> ConsensusAnalyzer -> PPOAgent (Arbiter) -> Executor -> MT5
```

### Key Architectural Components

1. **SmartInfoBus** (`modules/utils/info_bus.py`): Singleton message bus with ownership tracking
   ```python
   bus = InfoBusManager.get_instance()
   bus.set("key", value, module="ModuleName", thesis="explanation")
   bus.get("key", "ConsumerModule", default=None)
   ```

2. **Module Contracts** (`modules/contracts.py`): **Source of truth** for all module dependencies
   - Every module's `provides`/`requires` arrays are defined here
   - Single-writer ownership enforced (see comments for canonical owners)

3. **PPOAgent v3.0** (`modules/meta/`): **3-Layer Architecture**
   - **PPOCore** (`ppo_core.py`): Pure RL engine - action selection, GAE, PPO update
   - **ArbiterLogic** (`arbiter_logic.py`): Per-instrument decisions, gating, explanations
   - **PPOAgentShell** (`ppo_agent_shell.py`): SmartInfoBus gateway, lifecycle management
   
   Outputs:
   - `ppo_final_decision`: Primary instrument decision
   - `ppo_multi_decision`: Dict with per-instrument decisions
   - `ppo_gate_passed`, `ppo_position_size`: For Executor
   - `ppo_instrument_stats`: Per-instrument statistics

4. **Unified Voting (v5.0)** (`modules/voting/`): Modular pipeline
   - Experts: `ThemeExpert`, `TrendExpert`, `MomentumExpert`, `SeasonalityRiskExpert`
   - Stages: `CommitteeCoordinator` -> `ConsensusAnalyzer` -> `CollusionDetector` -> `FinalArbiter`
   - Kernel: `SlimVotingKernel` orchestrates the pipeline

## PPO Agent Architecture (v3.0)

```
┌──────────────────────────────────────────────────────────────┐
│  PPOAgentShell (SmartInfoBus Gateway)                        │
│  - Gathers signals from bus                                   │
│  - Publishes decisions to bus                                 │
│  - Manages lifecycle & health                                 │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  ArbiterLogic (Domain Logic)                                  │
│  - Per-instrument observation building                        │
│  - Gating pipeline (risk + memory)                           │
│  - InstrumentDecision creation                                │
│  - Explanation generation                                     │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│  PPOCore (Pure RL Engine)                                     │
│  - EnhancedPPONetwork (actor-critic)                         │
│  - select_action(obs) → (action, log_prob, value)            │
│  - record_step() / update() for training                      │
│  - NO SmartInfoBus knowledge                                  │
└──────────────────────────────────────────────────────────────┘
```

### Key Types (`ppo_types.py`)

```python
from modules.meta.ppo_types import InstrumentDecision, ArbiterMultiDecision

# InstrumentDecision fields:
#   instrument, direction, confidence, position_size, trust_score,
#   committee_action, expert_consensus, regime, gate_passed, reasoning, meta

# ArbiterMultiDecision fields:
#   instruments: Dict[str, InstrumentDecision]
#   global_meta: Dict[str, Any]
```

### Observation Builder (v4.0)

```python
from modules.meta.ppo_observation_builder import (
    build_ppo_observation,                   # Global observation
    build_ppo_observation_for_instrument,    # Per-instrument observation
    PPO_OBS_SIZE,                           # 64 dims (v4.0)
    FEATURE_GROUPS,                         # Feature group indices
)

# Feature groups (v4.0):
# - market (0-16): prices, volume, volatility
# - account (16-24): balance, exposure, positions
# - risk (24-32): risk levels, drawdown, memory gate
# - committee (32-40): consensus, expert signals
# - regime (40-48): market regime, trends
# - world_model (48-56): price/volatility predictions, scenarios
# - trading_mode (56-64): mode constraints, risk multipliers
```

### Integration Info Types

```python
from modules.meta.arbiter_logic import (
    StrategyInfo,        # BiasAuditor + CurriculumPlanner + ThesisEvolution
    TradingModeInfo,     # TradingModeManager position/risk constraints
    WorldModelInfo,      # EnhancedWorldModel predictions
)
```

## Module Template

```python
from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager

@module(**module_args("ModuleName"))  # Pulls contracts from contracts.py
class MyModule(BaseModule):
    def _initialize(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()
    
    async def process(self, **inputs) -> Dict[str, Any]:
        # MUST return dict with all keys from self.metadata.provides
        return {"output_key": value, "_thesis": "explanation"}
```

**Critical Rules:**
- Single-writer ownership: each bus key has ONE canonical provider
- When adding outputs: update `modules/contracts.py` AND `config/module_registry.yaml`
- `_thesis` required for explainable modules (`thesis_required=True`)
- Voting members must provide `{ModuleName}_voting_proposal` and `{ModuleName}_confidence`

## Key Commands

```bash
# Dashboard (backend + frontend)
python run_dashboard.py --dev          # Development mode
python run_dashboard.py --prod         # Production mode

# Training
python train/train_ppo_hybrid.py --mode offline --timesteps 100000
python train/train_ppo_hybrid.py --mode online --timesteps 50000

# Live trading
python start_live_trading.py --symbol XAUUSD
```

## Configuration Files

| File | Purpose |
|------|---------|
| `config/system_config.yaml` | Module timeouts, execution settings |
| `config/module_registry.yaml` | Auto-generated module metadata |
| `config/risk_policy.yaml` | Prop firm rules, lot sizing, risk limits |
| `modules/contracts.py` | **Source of truth** for module dependencies |

## Project Conventions

### Observation Schema
Training and live use **identical** 64-dim observations (`PPO_OBS_SIZE` v4.0). Never change dimensions without updating both `ppo_observation_builder.py` and `envs/modern_env.py`.

### Error Handling
```python
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
self.error_handler = create_error_handler("ModuleName", ErrorPinpointer())
```

### Logging
```python
from modules.utils.audit_utils import RotatingLogger
self.logger = RotatingLogger("ModuleName", log_path="logs/category/module.log", operator_mode=True)
```

### Bus Data Access with Fallbacks
```python
regime = self.smart_bus.get("market_regime", self.__class__.__name__, default="UNKNOWN")
```

## File Organization

```
modules/
├── contracts.py          # Source of truth for dependencies
├── core/                 # BaseModule, Orchestrator, mixins
├── voting/               # v5.0 pipeline (experts/, stages/, pipeline/)
├── meta/                 # PPO Agent architecture
│   ├── ppo_types.py      # Decision dataclasses
│   ├── ppo_core.py       # Pure RL engine
│   ├── arbiter_logic.py  # Domain logic + StrategyInfo/TradingModeInfo/WorldModelInfo
│   ├── ppo_agent_shell.py # SmartInfoBus gateway
│   ├── ppo_agent.py      # Legacy monolithic (for reference)
│   └── ppo_observation_builder.py  # v4.0 observation builder (64 dims)
├── risk/                 # DynamicRiskController, PortfolioRiskSystem
├── memory/               # UnifiedMemory (provides memory_gate, danger_zones)
├── executor/             # Executor (final trade execution)
└── external/             # MarketDataProvider

envs/
├── modern_env.py         # Gym environment (uses PPO_OBS_SIZE)
└── config.py             # TradingConfig, MarketState

train/
└── train_ppo_hybrid.py   # Main training script
```

## Debugging Tips

- **Circuit breakers**: Check `config/system_config.yaml` for `circuit_breaker_thresholds`
- **InfoBus data flow**: Logs in `logs/infobus/`
- **Stale data warnings**: Increase `adaptive.stale_warn_s` in system_config or fix producer
- **Module timeouts**: Adjust `execution.timeouts.by_module` in system_config
- **PPO decisions**: Check `ppo_multi_decision` for per-instrument reasoning
