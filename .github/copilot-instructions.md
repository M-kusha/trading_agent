# AI Trading System - Copilot Instructions

## Architecture Overview

This is a **PPO-Lagrangian reinforcement learning trading system** built around a modular, event-driven architecture called **SmartInfoBus**. The system trades forex pairs (EUR/USD, XAU/USD) across multiple timeframes (H1, H4, D1).

### Core Architectural Concepts

1. **SmartInfoBus** (`modules/utils/info_bus.py`): Centralized message bus for inter-module communication. Modules publish/subscribe to named keys with ownership tracking.
   - Use `InfoBusManager.get_instance()` to access the singleton bus
   - `bus.set(key, value, module="ModuleName", thesis="explanation")` to publish
   - `bus.get(key, "ConsumerModule", default=None)` to consume

2. **Module Contracts** (`modules/contracts.py`): Each module declares `provides` and `requires` arrays defining data flow. This is the single source of truth for understanding module dependencies.

3. **ModuleOrchestrator** (`modules/core/module_system.py`): Executes modules in topologically-sorted order based on contracts. Handles circuit breakers, timeouts, and health monitoring.

4. **BaseModule** (`modules/core/module_base.py`): All modules inherit from this and use the `@module` decorator with contract metadata.

### Module Categories

| Category | Purpose | Key Modules |
|----------|---------|-------------|
| `external` | Data ingestion | `MarketDataProvider`, `SessionManager` |
| `features` | Feature engineering | `AdvancedFeatureEngine`, `MultiScaleFeatureEngine` |
| `market` | Market analysis | `UnifiedMarketModule` |
| `voting` | Decision consensus | `VotingKernel`, `StrategyArbiter`, `CollusionAuditor` |
| `risk` | Risk management | `DynamicRiskController`, `PortfolioRiskSystem` |
| `memory` | Pattern memory | `UnifiedMemory` |
| `executor` | Trade execution | `Executor`, `PositionManager` |
| `meta` | Meta-learning | `PPOAgent`, `MetaAgent` |

## Creating/Modifying Modules

### Module Template
```python
from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager

@module(**module_args("ModuleName"))  # Pulls contracts from modules/contracts.py
class MyModule(BaseModule):
    def _initialize(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()
        # Setup state, loggers, config
    
    async def process(self, **inputs) -> Dict[str, Any]:
        # Main processing logic
        # MUST return dict with all keys from self.metadata.provides
        return {"output_key": value, "_thesis": "explanation"}
```

### Contract Rules (Critical)
- **Single-writer ownership**: Each bus key has ONE canonical provider (see comments in `contracts.py`)
- When adding new outputs: update `modules/contracts.py` AND `config/module_registry.yaml`
- `_thesis` key required for explainable modules (`thesis_required=True`)
- Voting members must provide `{ModuleName}_voting_proposal` and `{ModuleName}_confidence`

## Key Development Workflows

### Running the System
```bash
# Full dashboard (backend + frontend dev server)
python run_dashboard.py --dev

# Backend only (production)
python run_dashboard.py --prod

# Training (offline mode with CSV data)
python train/train_ppo_hybrid.py --mode offline --timesteps 100000

# Training (online with MT5)
python train/train_ppo_hybrid.py --mode online --timesteps 50000
```

### Testing Module Changes
1. Verify contracts: Check `modules/contracts.py` for consistency
2. Run orchestrator in debug: Set `debug=True` in `config/system_config.yaml`
3. Monitor InfoBus: Check `logs/infobus/` for data flow issues
4. Watch circuit breakers: Modules auto-disable after repeated timeouts

### Configuration Hierarchy
1. `config/system_config.yaml`: Module timeouts, execution settings, per-module configs
2. `config/module_registry.yaml`: Module paths, provides/requires (auto-generated from contracts)
3. `config/risk_policy.yaml`: Trading risk limits and constraints

## Common Patterns

### Consuming Bus Data with Fallbacks
```python
market_regime = self.smart_bus.get("market_regime", self.__class__.__name__, default="UNKNOWN")
```

### Publishing Coordination IDs (Voting Pipeline)
```python
# Generate decision coordination ID
decision_id = f"{datetime.now().isoformat()}#{tick_count}"
self.smart_bus.set("decision_id", decision_id, module="VotingKernel", thesis="Coordination ID")
```

### Error Handling Pattern
```python
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler

self.error_pinpointer = ErrorPinpointer()
self.error_handler = create_error_handler("ModuleName", self.error_pinpointer)
```

### Using Rotating Loggers
```python
from modules.utils.audit_utils import RotatingLogger, format_operator_message

self.logger = RotatingLogger(
    name="ModuleName",
    log_path="logs/category/module_name.log",
    max_lines=5000,
    operator_mode=True,
    plain_english=True,
)
```

## Important Constraints

- **Timeouts**: All modules have configurable timeouts in `system_config.yaml`. Respect the `timeout_ms` from metadata.
- **Bus key naming**: Use namespaced keys to avoid conflicts (e.g., `kernel_decision_id`, `consensus_score`)
- **Live vs Sim mode**: Check `bus.get("execution_mode")` before executing real trades
- **Memory modules**: `UnifiedMemory` provides `memory_gate`, `danger_zones`, `playbook_recall` signals consumed by risk/voting

## File Organization

```
modules/
├── contracts.py          # Source of truth for module dependencies
├── core/                 # Framework (BaseModule, Orchestrator, mixins)
├── voting/              # VotingKernel orchestrates all voting modules
├── risk/                # Risk controllers and monitors
├── memory/              # UnifiedMemory consolidates all memory subsystems
├── executor/            # Trade execution (Executor is the final gate)
└── external/            # Data providers (MarketDataProvider publishes market_data)

config/
├── system_config.yaml    # Per-module timeouts and settings
└── module_registry.yaml  # Auto-generated module metadata

train/
├── train_ppo_hybrid.py   # Main training entry point
└── enhanced_training_callback.py  # Telemetry during training
```
