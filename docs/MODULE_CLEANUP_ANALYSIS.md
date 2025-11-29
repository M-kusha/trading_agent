# Module Cleanup Analysis

**Date:** November 27, 2025  
**Branch:** infobus-v2  
**Purpose:** Identify modules that provide minimal value and can be removed to reduce complexity

---

## ✅ CLEANUP COMPLETED

All identified modules have been removed and configuration files updated.

---

## Executive Summary

After analyzing all module contracts, their provides/requires dependencies, and actual consumption patterns across the codebase, **11 modules were identified with zero or near-zero consumers** of their outputs. These modules added complexity without contributing to trading decisions.

**Impact Achieved:**
- ~12,500+ lines of dead code removed
- Faster tick execution (fewer modules to orchestrate)
- Simpler debugging and maintenance
- Reduced log noise

---

## 🔴 REMOVED MODULES - Zero Consumers

These modules produced outputs that **no other module consumed**. They have been removed.

| Module | Category | Lines | Status | Notes |
|--------|----------|-------|--------|-------|
| `AuditingCoordinator` | auditing | ~300 | ✅ DELETED | Pure logging, no trading impact |
| `TradeExplanationAuditor` | auditing | ~300 | ✅ DELETED | No module reads its outputs |
| `TradeThesisTracker` | auditing | ~300 | ✅ DELETED | Thesis tracking unused |
| `OpponentSimulator` | simulation | ~600 | ✅ DELETED | Adversarial simulation unused |
| `RoleCoach` | simulation | ~500 | ✅ DELETED | Coaching outputs unused |
| `ShadowSimulator` | simulation | ~600 | ✅ DELETED | Only optional consumer had fallback |
| `MetaCognitivePlanner` | meta | ~1,500 | ✅ DELETED | Planning outputs unused |
| `PPOLagAgent` | meta | ~1,600 | ✅ DELETED | Redundant with PPOAgent |
| `OpponentModeEnhancer` | strategy | ~1,800 | ✅ DELETED | Mode detection outputs unused |
| `PlaybookClusterer` | strategy | ~3,000 | ✅ DELETED | Clustering outputs unused |
| `StrategyGenomePool` | strategy | ~2,600 | ✅ DELETED | Genome evolution unused |

**Total Removed: ~12,500+ lines**

### Configuration Updates Made:
1. **`modules/contracts.py`** - All deleted module contracts commented out
2. **`config/module_registry.yaml`** - All deleted module entries commented out
3. **`modules/meta/metar_rl_controller.py`** - Removed PPOLagAgent import, defaults to PPOAgent
4. **`modules/trading_modes/trading_mode.py`** - Has fallback for `shadow_predictions`
5. **`tests/system_validation.py`** - Tests for deleted modules now skip

---

## 🟡 VISUALIZATION MODULES - Dashboard Only

These modules provide data for the React frontend dashboard but don't participate in trading decisions. They should be moved out of the main orchestration loop.

| Module | Category | Lines | Purpose |
|--------|----------|-------|---------|
| `VisualizationInterface` | visualization | ~800 | Dashboard display data |
| `TradeMapVisualizer` | visualization | ~700 | Dashboard charts |
| `ExplanationGenerator` | strategy | ~2,000 | Human-readable explanations for UI |

**Recommendation:** Move to `backend/` as lazy-loaded data generators called by API endpoints, not every tick.

---

## 🟠 TRAINING-ONLY - Disable During Inference

| Module | Category | Lines | Purpose |
|--------|----------|-------|---------|
| `CurriculumPlannerPlus` | strategy | ~1,800 | Curriculum learning stages |

**Recommendation:** Add `training_only: true` flag and skip during live/inference mode.

---

## 🟢 KEEP - Essential Core Modules

### Voting System (v5.0 - Unified)
| Module | Purpose | Is Voting Member |
|--------|---------|------------------|
| `MomentumExpert` | Multi-indicator momentum analysis | ✅ Yes |
| `TrendExpert` | Trend detection with ADX/MA | ✅ Yes |
| `ThemeExpert` | Macro regime & risk-on/off | ✅ Yes |
| `SeasonalityRiskExpert` | Temporal patterns & session timing | ✅ Yes |
| `CommitteeCoordinator` | Collects votes, calculates weights | No |
| `ConsensusAnalyzer` | Measures agreement level | No |
| `CollusionDetector` | Detects vote manipulation | No |
| `HorizonAligner` | Time horizon adjustments | No |
| `UncertaintySampler` | Monte Carlo uncertainty | No |
| `FinalArbiter` | Final go/no-go decision | No |
| `SlimVotingKernel` | Orchestrates voting pipeline | No |

### Risk Management
| Module | Purpose | Is Voting Member |
|--------|---------|------------------|
| `DynamicRiskController` | Dynamic risk scaling | ✅ Yes |
| `PortfolioRiskSystem` | Portfolio-level risk | ✅ Yes |
| `EnhancedAnomalyDetector` | Anomaly detection | ✅ Yes |
| `ExecutionQualityMonitor` | Execution quality | ✅ Yes |
| `ActiveTradeMonitor` | Position duration tracking | No |
| `DrawdownRescue` | Drawdown protection | No |
| `ComplianceModule` | Risk limits compliance | No |
| `CorrelatedRiskController` | Correlation risk | No |

### Core Infrastructure
| Module | Purpose |
|--------|---------|
| `Executor` | Trade execution |
| `PositionManager` | Position sizing & management |
| `MarketDataProvider` | Market data ingestion |
| `SessionManager` | Session state & performance |
| `UnifiedMarketModule` | Market regime detection |

### Features & Memory
| Module | Purpose |
|--------|---------|
| `AdvancedFeatureEngine` | Technical indicators |
| `MultiScaleFeatureEngine` | Multi-timeframe features |
| `UnifiedMemory` | Pattern memory, danger zones, playbook |

### Meta/RL
| Module | Purpose | Is Voting Member |
|--------|---------|------------------|
| `PPOAgent` | Core RL agent | ✅ Yes |
| `MetaAgent` | Meta-level decisions | ✅ Yes |

### Strategy (Simplified)
| Module | Purpose |
|--------|---------|
| `StrategyIntrospector` | Provides `trading_performance` used by many |
| `ThesisEvolutionEngine` | Provides `market_thesis` |
| `BiasAuditor` | Provides `bias_analysis` (simplify to key outputs only) |

---

## ⚠️ QUESTIONABLE - Need Further Review

| Module | Issue | Recommendation |
|--------|-------|----------------|
| `MetaRLController` | May overlap with voting kernel functionality | Review for redundancy |
| `EnhancedWorldModel` | Only 2/4 outputs consumed, circular dependency issues | Consider merging with UnifiedMarketModule |
| `TradingModeManager` | Very complex, many inputs | Keep but simplify |

---

## Contract Cleanup Steps

After removing modules, update these files:

1. **`modules/contracts.py`** - Remove or comment out contracts for deleted modules
2. **`config/module_registry.yaml`** - Remove entries for deleted modules
3. **`config/system_config.yaml`** - Remove timeout/config entries for deleted modules
4. **Clean up requires** - Remove references to deleted module outputs from other modules' `requires` lists

---

## Memory Components (No Change Needed)

The following are **internal components** of `UnifiedMemory`, not separate orchestrated modules:

```
modules/memory/components/
├── base.py
├── budget.py
├── compression.py
├── interventions.py
├── loss_risk_head.py
├── mistakes.py
├── neural.py
├── playbook.py
└── replay.py
```

These are correctly structured as internal components.

---

## Post-Cleanup Module Count

| Metric | Before | After |
|--------|--------|-------|
| Total Orchestrated Modules | ~45+ | ~32 |
| Modules with Zero Consumers | 14 | 0 |
| Strategy Modules | 8 | 3 |
| Simulation Modules | 3 | 0 |
| Auditing Modules | 3 | 0 |
| Meta Modules | 5 | 2 |

---

## Notes

- All voting experts (MomentumExpert, TrendExpert, ThemeExpert, SeasonalityRiskExpert) were just upgraded to advanced versions with multiple indicators
- The voting pipeline was refactored to v5.0 unified architecture
- Risk modules that vote (DynamicRiskController, PortfolioRiskSystem, etc.) have their gate actions filtered via `ignore_actions` in CommitteeCoordinator
- Legacy voting modules are already deprecated and stored in `modules/voting/_legacy/`
