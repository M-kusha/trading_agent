SmartInfoBus Refactor — Contract and Wiring Corrections

Summary
- Integrated canonical system_health into HealthMonitor and publish a consolidated snapshot on the bus.
- Made TradingModeManager the owner and single writer of performance_data; published a consistent schema used downstream.
- Resolved duplicate providers of instrument_signals by removing PositionManager as a provider and namespacing its internal signals.
- Removed legacy modules/voting_1/* from the codebase (unregistered and deleted).
- Synced module registry entries for PositionManager and TradingModeManager with contracts.
- Kept existing invariants for reward shaping, circuit breakers, and error handling intact.

Changes
1) System Health (canonical)
   - File: modules/monitoring/health_monitor.py
     - Register HealthMonitor as provider of 'system_health'.
     - Publish 'system_health' in the publisher loop with structure:
       { overall_status, system, modules, performance, checks_performed, errors_encountered, timestamp }.

2) Performance Data (owner = TradingModeManager)
   - File: modules/trading_modes/trading_mode.py
     - After computing performance_data, publish to bus under key 'performance_data' with schema:
       { performance_data: <flat metrics>, environment_config: <dict> }.
     - Updated contract in modules/contracts.py so TradingModeManager provides 'performance_data'.

3) Duplicate Providers (instrument_signals)
   - File: modules/contracts.py
     - Removed 'instrument_signals' from PositionManager provides.
   - File: modules/position/position.py
     - Stopped publishing 'instrument_signals'.
     - Namespaced PM’s signals to 'position_manager_instrument_signals' for diagnostics.
   - StrategyArbiter remains sole owner/provider of 'instrument_signals'.

4) Legacy Cleanup
   - Removed modules/voting_1/* tree (unused, not in orchestrator discovery).

5) Registry Hygiene
   - File: config/module_registry.yaml
     - Synced PositionManager provides/requires to contracts.
     - Added TradingModeManager with up-to-date provides (including performance_data) and requires.

Non‑changes (explicit)
- Reward math, clipping ranges, circuit-breaker semantics, and error/fallback behavior left unchanged.
- Orchestrator planning and execution logic unchanged, aside from being consistent with the updated contracts.

Notes for Owners
- system_health single writer: HealthMonitor.
- performance_data single writer: TradingModeManager.
- instrument_signals single writer: StrategyArbiter.
- shaped_reward single writer: RiskAdjustedReward.
- risk_data/risk_score single writer: PortfolioRiskSystem.
- trade_vote single writer: EnhancedVotingCommitteeCoordinator.

Validation Checklist
- No duplicate providers for instrument_signals.
- All modules requiring performance_data or system_health now have valid providers.
- Contracts (modules/contracts.py) are consistent with config/module_registry.yaml for edited modules.
