# Module Cleanup Analysis

**Date:** November 27, 2025  
**Branch:** infobus-v2  
**Purpose:** Identify modules that provide minimal value and can be removed to reduce complexity

---

## Executive Summary

After analyzing all module contracts, their provides/requires dependencies, and actual consumption patterns across the codebase, **14 modules were identified with zero or near-zero consumers** of their outputs. These modules add complexity without contributing to trading decisions.

**Estimated Impact:**
- ~12,500+ lines of dead code can be removed
- Faster tick execution (fewer modules to orchestrate)
- Simpler debugging and maintenance
- Reduced log noise

---

## 🔴 IMMEDIATE REMOVAL - Zero Consumers

These modules produce outputs that **no other module consumes**. They can be safely removed with no impact on trading functionality.

| Module | Category | Lines | Outputs Provided | Consumers | Reason |
|--------|----------|-------|------------------|-----------|--------|
| `AuditingCoordinator` | auditing | ~300 | 3 | 0 | Pure logging, no trading impact |
| `TradeExplanationAuditor` | auditing | ~300 | 3 | 0 | No module reads its outputs |
| `TradeThesisTracker` | auditing | ~300 | 2 | 0 | Thesis tracking unused |
| `OpponentSimulator` | simulation | ~600 | 9 | 0 | Adversarial simulation unused |
| `RoleCoach` | simulation | ~500 | 9 | 0 | Coaching outputs unused |
| `ShadowSimulator` | simulation | ~600 | 8 | 1 (optional) | Only `shadow_predictions` used by TradingModeManager with fallback |
| `MetaCognitivePlanner` | meta | ~1,500 | 4 | 0 | Planning outputs unused |
| `PPOLagAgent` | meta | ~1,600 | 4 | 0 | Redundant with PPOAgent |
| `OpponentModeEnhancer` | strategy | ~1,800 | 7 | 0 | Mode detection outputs unused |
| `PlaybookClusterer` | strategy | ~3,000 | 8 | 0 | Clustering outputs unused |
| `StrategyGenomePool` | strategy | ~2,600 | 7 | 0 | Genome evolution unused |

**Total: ~12,500+ lines**

### Files to Remove:
```
modules/auditing/auditing_coordinator.py
modules/auditing/trade_explanation_auditor.py
modules/auditing/trade_thesis_tracker.py
modules/simulation/opponent_simulator.py
modules/simulation/role_coach.py
modules/simulation/shadow_simulator.py
modules/meta/metacognitive_planner.py
modules/meta/ppo_lag_agent.py
modules/strategy/opponent_mode_enhancer.py
modules/strategy/playbook_clusterer.py
modules/strategy/strategy_genome_pool.py
```

---

## 🟡 MOVE TO DASHBOARD BACKEND - Not Core Trading

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
