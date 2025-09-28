# Legacy Files Backup

This directory contains the old, fragmented system files that have been replaced by the unified architecture.

## ⚠️ **DEPRECATED FILES**

**These files are kept for reference only and should NOT be used in new development.**

## What's in Here

### Environment Files (`envs/`)
- `modern_env.py` - Old fragmented trading environment
- `env.py` - Legacy compatibility wrapper
- `config.py` - Old configuration system (dataclass-based)
- `shared_utils.py` - Legacy utility functions

### Training Files (`train/`)
- `train_ppo_hybrid.py` - Old training script with argument parsing issues
- `enhanced_training_callback.py` - Legacy training callbacks

### Live Trading Files (`live/`)
- `live_connector.py` - Old MT5 connector with fragmented patterns
- `mt5_credentials.py` - Legacy credential management

### Documentation
- `MEMORY_INTEGRATION.md` - Legacy module integration docs
- `RISK_INTEGRATION.md` - Legacy risk system docs
- `VOTING_INTEGRATION.md` - Legacy voting system docs
- `voting_components_continuation.txt` - Legacy notes

### Test Files
- `test_*.py` - Legacy test files for old architecture

### Scripts
- `dashboard.js` - Legacy dashboard
- `smart_setup.py` - Legacy setup script
- `run_dashboard.py` - Legacy dashboard runner

## 🚀 **Use the New Unified System Instead**

### Old vs New Mapping

| **Old (Deprecated)** | **New (Unified)** |
|---------------------|-------------------|
| `envs.modern_env.ModernTradingEnv` | `envs.unified_trading_env.UnifiedTradingEnv` |
| `envs.config.TradingConfig` | `config.unified_config.UnifiedConfig` |
| `train.train_ppo_hybrid` | `train.unified_training` |
| `live.live_connector` | `live.unified_live_trading` |

### Migration Examples

**Old Environment:**
```python
from envs.modern_env import ModernTradingEnv
from envs.config import TradingConfig

config = TradingConfig()
env = ModernTradingEnv(data_dict, config)
```

**New Environment:**
```python
from envs.unified_trading_env import create_trading_environment
from config.unified_config import ConfigFactory

config = ConfigFactory.create("development")
env = create_trading_environment(config)
```

**Old Training:**
```bash
python train/train_ppo_hybrid.py --mode offline --timesteps 100000
```

**New Training:**
```bash
python train/unified_training.py --preset development --timesteps 100000
```

## 🗑️ **Future Cleanup**

These files will be removed in a future version once the migration is complete and all references have been updated.

## 📚 **Documentation**

See `UNIFIED_ARCHITECTURE.md` in the project root for complete documentation of the new unified system.

---
**Last Updated:** $(date)
**Status:** DEPRECATED - DO NOT USE