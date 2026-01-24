# Voting System Audit + Module Map (v5.x)

This file is the single source of truth for:
- the `modules/voting/` package map + architecture
- contracts/bus-key surfaces (what is produced/consumed)
- PPO observation integration (what must exist for strict obs builds)
- duplication + legacy cleanup plan

---

## 1) Module Tree (authoritative)

```
modules/voting/
  __init__.py
  core/
    __init__.py
    base.py
    constants.py
    dynamic_thresholds.py
    per_instrument.py
    types.py
  experts/
    __init__.py
    base.py
    trend.py
    momentum.py
    theme.py
    seasonality.py
  stages/
    __init__.py
    committee.py
    consensus.py
    collusion.py
    horizon.py
    uncertainty.py
    arbiter.py
  pipeline/
    __init__.py
    kernel.py
  utils/
    __init__.py
    validators.py
    metrics.py
```

**High-level data flow**

`experts/*` → `stages/committee.py` → (`stages/consensus.py`, `stages/collusion.py`, `stages/horizon.py`, `stages/uncertainty.py`) → `stages/arbiter.py` → `pipeline/kernel.py`

---

## 2) Architecture Review (quality + fit)

### What’s strong
- **Clear layering**: Experts emit proposals → stages refine/score → arbiter gates → kernel aggregates/publishes.
- **Shared infrastructure**: `core/base.py` centralizes SmartInfoBus access, logging, performance tracking, circuit semantics.
- **Expert contract enforcement**: `experts/base.py` normalizes per-instrument outputs and provides a resilient process wrapper.
- **Backward-compat surfaces**: Stage modules and experts support older key names where needed (useful while migrating).

### What needs attention (architecture hygiene)
- **Compatibility keys are mixed with canonical keys** (good short-term, but creates drift long-term). Plan below.
- **Some modules still call `smart_bus.get/set` directly** instead of `VotingModuleBase.bus_get/bus_set`. It’s not wrong, but it creates inconsistent error handling and makes future bus changes harder.

---

## 3) Canonical “Single Source of Truth”

### Instrument normalization
- Canonical normalizer: `modules/voting/core/constants.py` → `normalize_instrument()`
- Usage rule: do not implement local normalizers in experts/stages. Always call the canonical one (directly or via `VotingModuleBase.canon()`).

### Vote/pipeline types
- Canonical types: `modules/voting/core/types.py` (`VotingProposal`, `VoteBundle`, `ConsensusResult`, `CollusionResult`, helpers)
- Per-instrument aggregation types only: `modules/voting/core/per_instrument.py` (`InstrumentProposal`, `PerInstrumentVote`, `aggregate_all_instruments`, `extract_instrument_data`)

---

## 4) Contracts & SmartInfoBus Key Surfaces

### Expert outputs (canonical)
Each expert should publish:
- `{ExpertClass}_voting_proposal` (dict proposal)
- `{ExpertClass}_confidence` (float 0..1)
- `{ExpertClass}_per_instrument_votes` (dict of `instrument -> {action/confidence/...}`)

### Expert analysis surfaces (required by PPO strict observation)
`modules/meta/ppo_observation_builder.py` strict mode requires these analysis keys:
- `trend_analysis` with `per_instrument["XAUUSD"]` containing required trend fields
- `momentum_analysis` with `per_instrument["XAUUSD"].rsi` present
- `theme_analysis` with `per_instrument["XAUUSD"].risk_regime` and `.volatility_regime` present
- `seasonality_risk_analysis` with `per_instrument["XAUUSD"]` present

**Important**: even “neutral/flat” fallbacks must still include the `"XAUUSD"` per-instrument block, otherwise strict observation building hard-fails.

### Orchestrator-enforced outputs (ModuleRegistry)
These keys are validated by `modules/core/module_base.py::validate_outputs()` against `config/module_registry.yaml`.

- `ThemeExpert` must output: `theme_volatility_regime`, `theme_trend_regime`, `theme_risk_regime`, `theme_composite_score`, `agreement_score`, `theme_expert_analysis`, `theme_expert_thesis`
- `SeasonalityRiskExpert` must output: `seasonal_session`, `seasonal_dow_bias`, `seasonal_monthly_pattern`, `seasonal_composite_score`, `seasonal_rollover_risk`, `seasonal_weekend_risk`, `seasonality_analysis`, `seasonality_expert_analysis`, `seasonality_expert_thesis`

### Committee outputs
Produced by `stages/committee.py`:
- `committee_decision`, `committee_confidence`
- `committee_votes`, `expert_votes`, `raw_proposals`, `member_confidences`, `voting_weights`
- `committee_decisions_by_instrument` (authoritative per-symbol decision surface)

### Downstream stage outputs
- Consensus: `consensus_result`, `consensus_score`, `agreement_score`
- Collusion: `collusion_result`, `collusion_score`, `collusion_detected`
- Horizon: `horizon_alignment`, `aligned_weights`
- Uncertainty: `uncertainty_result`, `fragility`, `instrument_fragility`
- Final arbiter: `final_decision`, `gate_decision`, `trade_vote`
- Kernel: `trade_vote_v2`, `kernel_decision`, `kernel_consensus_score`, `kernel_instrument_signals`

---

## 5) PPO Integration Audit (what uses votes, and how)

### 5.1 PPOAgentShell ↔ voting system (runtime)
`modules/meta/ppo_agent_shell.py` consumes voting outputs:
- Reads `committee_decisions_by_instrument`, `committee_decision`, `committee_confidence`, `consensus_score`, `fragility`
- Reads expert proposals via `{Expert}_voting_proposal` + `{Expert}_confidence`

And produces a vote back into the committee:
- Publishes `PPOAgent_voting_proposal` and `PPOAgent_confidence` (so PPO can be a committee member)

This is a **hybrid architecture**:
- PPO is the primary “intelligent arbiter”
- The committee/voting system provides consensus + fragility context and can be used as a fallback path elsewhere

### 5.2 PPO observation (training + live parity)
`modules/meta/ppo_observation_builder.py` builds the PPO observation vector from SmartInfoBus keys, including:
- expert signals (Trend/Momentum/Theme/Seasonality)
- committee state (decision/confidence/consensus_score/fragility)
- risk/memory/world-model/timing/governor state

**Rule**: If you change what the observation vector contains (layout/dimension), you must:
- bump `PPO_OBS_VERSION` / `PPO_OBS_SIZE`
- retrain the PPO model

**Safe upgrades** (no retrain required):
- improving the upstream expert computations as long as the same keys exist and types stay consistent
- adding internal/diagnostic keys that are NOT consumed by `ppo_observation_builder.py`

---

## 6) Recommendation: Keep voting, or remove it?

### Option A (recommended default): Keep hybrid (PPO primary + voting fallback + metrics)
Pros:
- robustness when PPO observation fails (committee + `trade_vote_v2` remain usable)
- interpretable safety signals (`consensus_score`, `fragility`, collusion) remain available for logging/guards/reward shaping
- avoids a large breaking migration across PositionManager + Reward + Monitoring

Cons:
- more moving parts
- PPO participates in the committee, which can create a mild “feedback loop” (committee contains PPO vote; PPO observes committee)

### Option B: Remove voting entirely (experts only feed PPO)
This is a **system-wide change**, not just `modules/voting/`.

If you remove voting, you must also update (minimum):
- `modules/position/position_logic.py` and `modules/position/position_base.py` (remove committee fallback + `trade_vote_v2` usage)
- `modules/reward/components/data_extractor.py` (removes `kernel_consensus_score`/`consensus_score` usage or replace with PPO-only metrics)
- `modules/monitoring/system_integrity_suite.py` (stop expecting voting keys)
- `modules/meta/ppo_observation_builder.py` (remove committee-state requirement OR replace it with a PPO-local “committee-like” aggregation)
- `config/module_registry.yaml` and `modules/contracts.py` (remove/disable voting modules and their provides/requires)

Practical warning:
- If you remove committee-state from the PPO observation, you must retrain due to observation layout changes.

### Option C (clean middle ground): Keep committee, drop kernel/arbiter execution dependency
If your goal is “experts are signal givers”, you can still keep:
- `CommitteeCoordinator` (and possibly `Consensus/Uncertainty`) purely as *signal processors*
and stop using:
- `FinalArbiter` + `SlimVotingKernel` as “trade signal producers”

This keeps the valuable consensus/fragility metrics without keeping the full voting “decision stack”.

---

## 7) Duplication & Legacy Cleanup Plan (non-breaking)

### Cleanups already applied (to reduce duplication)
- `modules/voting/core/per_instrument.py` is now scoped to per-instrument aggregation only (no duplicate vote/pipeline dataclasses).
- Removed duplicate local `normalize_instrument()` implementations in experts; everything routes through `core/constants.py`.

### Remaining legacy/compat surfaces (intentionally present)
- Alias keys like `trend_voting_proposal`, `theme_voting_proposal`, `seasonality_voting_proposal`, etc.
- Stage modules accept legacy shapes (e.g., `flat` vs `hold`, older proposal formats).

### Safe deprecation path (do not break production)
1) **Inventory consumers** (search for key usage) before deleting any alias keys.
2) Introduce a single config flag, e.g. `voting.publish_legacy_keys: true|false` (default `true`).
3) Turn the flag off in one environment first (training/staging), confirm no missing-key failures.
4) Remove alias-key publishing only after:
   - contracts/config are updated
   - monitoring suite expectations are updated
   - PPO obs strict builder still passes

---

## 8) If you want me to help with the “remove voting” migration

I can do this safely in phases:
1) Add a `config` toggle to disable voting modules at the orchestrator level (no deletions yet)
2) Update all consumers to not require voting keys (PositionManager/Reward/Monitoring/PPO obs)
3) Remove contracts/module_registry entries
4) Only then delete `modules/voting/` (optional)

---

## 9) Runtime Troubleshooting (logs)

### Symptom: `PPOAgentShell` warns about missing keys (ex: `TrendExpert_voting_proposal`, `trend_analysis`)
Most often: the producing module crashed and was disabled by the orchestrator, so the keys never refresh and become stale/absent to strict readers.

### Symptom: `CommitteeCoordinator` says "Collected 1 votes from 4 voters"
Most often: the other experts were disabled after repeated crashes, leaving only one expert producing fresh keys.

### What to check first
- `logs/orchestrator/orchestrator.log` for "missing required output: ...", "MODULE FAILED ...", and "Skipping disabled module ...".
- `config/module_registry.yaml` for each module's `provides:` list (those keys must be present in that module's returned output dict).
