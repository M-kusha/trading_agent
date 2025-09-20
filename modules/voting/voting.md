# Voting System — Architecture & Intelligence Audit (v1)
_Date: 2025-09-18 19:42 (local)_

This document captures **what you have today**, **how it operates end‑to‑end**, identified **gaps/risks**, and a concrete **upgrade plan to reach 10/10 intelligence**. It also proposes a **unified kernel** and a **reliable debugging timeline** so failures are easy to trace and fix.

---

## 1) What you have today (modules & roles)

### 1.1 `voting_wrappers.py`
**Roles**
- Defines `EnhancedVotingExpertBase` and concrete expert wrappers (e.g., `EnhancedThemeExpert`, `EnhancedSeasonalityRiskExpert`).  
- Provides `EnhancedVotingCommitteeCoordinator` to aggregate expert proposals, compute an internal consensus view, and publish a **canonical `trade_vote`** for downstream consumers.

**Key behaviors (observed in code)**
- Emits a consolidated `trade_vote` object with `action` and `confidence`, and publishes a `horizon_alignment` snapshot for context.
- Computes and attempts to publish a structured **`voting_consensus` (dict)** (e.g., `{consensus_exists, consensus_strength, per_expert_confidence}`). If it detects that the **owner** of `voting_consensus` is the `ConsensusDetector`, it falls back to **namespaced keys** and logs the fallback.
- Publishes **per‑expert audit info**: `committee_votes`, signals, timing/session hints, and performance feedback stubs.
- Uses SmartInfoBus mixins, `HealthMonitor`, and `PerformanceTracker` with safe fallbacks.

**Outputs seen in code**
- `trade_vote` (dict)
- `committee_votes` (list of per‑expert votes)
- `signals` (per‑instrument mapping)
- `horizon_alignment` (summary dict)
- Tries to publish `voting_consensus` (dict; see conflict in §3.1).

---

### 1.2 `consensus_detector.py`
**Role**
- Production‑grade consensus analysis for the committee.

**Key behaviors**
- Calculates **`consensus_score`** (float 0–1) using multiple methods (cosine agreement, direction alignment, confidence‑weighted measures, temporal smoothing & quality weighting).
- **Also writes `voting_consensus` (float alias)** == `consensus_score` for backward compatibility.
- Tracks performance and health; emits diagnostics and a human‑readable thesis.

**Outputs**
- `consensus_score` (float, canonical numeric consensus)
- `voting_consensus` (float alias; legacy surface)

**Inputs (implicit)**
- Voting actions & per‑member confidences (extracted from the bus/committee artifacts).

---

### 1.3 `collusion_auditor.py`
**Role**
- Detects anomalous coordination/“collusion” across expert votes.

**Key behaviors**
- Computes **`collusion_score`** (0–1) from similarities (cosine/correlation/euclidean), **temporal pattern** checks, and adaptive thresholds.
- Summarizes suspicious pairs and issues severity/recommendations.
- Tracks reliability: higher collusion → lower detection reliability.

**Outputs**
- `collusion_score` (float)
- Suspicious pairs / alerts (namespaced diagnostics)

**Inputs (implicit)**
- Voting actions over a rolling window; member identities; optionally confidences and volatility context.

---

### 1.4 `time_horizon_aligner.py`
**Role**
- Scales committee weights by horizon with **regime/session/performance** awareness.

**Key behaviors**
- Combines regime & session multipliers; returns normalized **`aligned_weights`** and structured **`horizon_alignment`** diagnostics.
- Publishes both to the bus; guards with neutral fallbacks when needed.

**Outputs**
- `aligned_weights` (list of floats)
- `horizon_alignment` (dict with factors/impact)

**Inputs (implicit)**
- `weights`, regime/session hints, performance feedback signals.

---

### 1.5 `strategy_arbiter.py`
**Role**
- Final gating and **per‑instrument signal** publication.

**Key behaviors**
- Builds a blended proposal from committee action + weights.
- Reads **`consensus_score`** and **`collusion_score`** to gate actions; tracks `gate_passes/attempts`, pass rate, and quality metrics.
- Publishes **`instrument_signals`** and keeps a rolling audit deque of gate decisions.

**Outputs**
- `instrument_signals` (mapping)
- Gating telemetry: pass rate, decisions, parameters.

**Inputs (implicit)**
- Blended action, `aligned_weights`, `consensus_score`, `collusion_score`, regime/session.

---

### 1.6 `alternative_reality_sampler.py`
**Role**
- Quantifies **uncertainty** by sampling alternative outcomes for robustness.

**Key behaviors**
- Generates structured perturbation/Monte‑Carlo‑like samples, computes an **uncertainty** estimate and suggests conservative posture when uncertainty exceeds threshold.
- Namespaces detailed diagnostics under `voting/ars/*` and logs to `logs/voting/alternative_reality_sampler.log`.

**Outputs**
- `uncertainty` (float estimate) and rich, namespaced diagnostics.

**Inputs (implicit)**
- Current vote weights/proposals; market uncertainty factor; sampler config.

---

## 2) How it operates today (end‑to‑end flow)

1) **Experts propose** actions & confidences via `EnhancedVotingExpertBase` derivatives.  
2) **Committee coordinator** aggregates them, computes an internal consensus view, and publishes:
   - `trade_vote` (with action & confidence)
   - `committee_votes`, session/volatility hints, and `horizon_alignment` summary
   - A structured `voting_consensus` (dict) **attempt** (falls back to namespaced keys if the consensus owner is different)
3) **ConsensusDetector** ingests votes/confidences and publishes `consensus_score` (and also a float alias `voting_consensus`).
4) **CollusionAuditor** runs similarity/temporal checks and publishes `collusion_score` + suspicious pairs.
5) **TimeHorizonAligner** transforms raw weights into `aligned_weights` and publishes a `horizon_alignment` explanation.
6) **StrategyArbiter** combines action + aligned weights, gates it through `consensus_score` & `collusion_score`, and emits `instrument_signals` with gate telemetry.
7) **AlternativeRealitySampler** estimates **uncertainty**, recommending a conservative stance when high.

All modules rely on SmartInfoBus, health/performance trackers, and emit op‑theses for observability.

---

## 3) Gaps / risks we should address

### 3.1 Conflicting meaning of `voting_consensus`
- **Today:** `ConsensusDetector` writes `voting_consensus` **as a float** (alias to `consensus_score`), while the **Committee** tries to write a **dict** under the same key. That’s a type clash that can confuse downstream readers.
- **Fix:** Deprecate `voting_consensus` entirely. Use:
  - **Numeric:** `consensus_score` (owner: ConsensusDetector)
  - **Structured:** `committee_consensus` (owner: Committee)
  - Keep a **temporary read‑only alias** for compatibility during migration.

### 3.2 Shape drift: `raw_proposals` / `votes`
- Some analytics expect **numeric lists** (vectors) while the committee publishes **rich dicts** for readability.  
- **Fix:** Committee additionally publishes **`proposal_vectors: List[List[float]]`** strictly for analytics, and **`member_confidences`** (list aligned to those vectors). Keep dicts for UI/logs.

### 3.3 Tick/race conditions
- Without a shared **`decision_id`** and `tick_ts`, modules may mix states across frames.
- **Fix:** Create & propagate `decision_id` from the committee; each stage tags outputs with it. Consumers only fuse values sharing the same `decision_id`.

### 3.4 Scope noise & key collisions
- Generic keys (e.g., performance or analytics fields) risk multi‑writer collisions.
- **Fix:** Namespace or prefix: `committee_*`, `consensus_*`, `collusion_*`, `tha_*`, `arb_*`, `ars_*`, or nest under a single **`voting/decision_bundle`** (see §4).

### 3.5 Missing learning loops
- No explicit **per‑expert reliability** or calibration feeding back into weights.
- No use of uncertainty to throttle position **size**.
- **Fix:** Add reliability (Brier/LogLoss) and **risk_throttle** derived from uncertainty.

---

## 4) Target “Voting Schema v1” (canonical contract)

> One bundle for downstream consumers + a few canonical single‑writer keys.

```jsonc
{
  "decision_id": "string",          // unique per tick
  "tick_ts": "ISO-8601",            // wall clock
  "committee": {
    "members": ["expA", "expB", "..."],
    "proposal_vectors": [[...], ...],      // numeric only
    "member_confidences": [0.0, ...],      // same order as vectors
    "committee_consensus": {
      "consensus_exists": true,
      "consensus_strength": 0.0
    },
    "raw": { "votes": [...], "meta": {...} } // optional rich dicts for UI
  },
  "consensus": { "score": 0.0, "components": {...} },
  "collusion": { "score": 0.0, "suspicious_pairs": [[i,j], ...], "pair_penalties": [[i,j,0.1], ...] },
  "weights": { "raw": [..], "aligned": [..] },
  "uncertainty": { "level": 0.0, "fragility": 0.0, "n_alts": 0 },
  "trade_vote_v2": { "action": "long|short|abstain", "size": 0.0, "horizon_minutes": 0, "confidence": 0.0, "reason": "..." },
  "signals": { "EURUSD": { "intensity": 0.0, "confidence": 0.0 } },
  "_schema_version": "v1"
}
```

**Canonical single‑writer keys on the bus**
- `trade_vote_v2` (flattened for PM)
- `consensus_score`
- `collusion_score`
- `aligned_weights`
- `signals`

---

## 5) Unified **VotingKernel** (one‑module debug surface)

> If you want a single module for easier debugging, wrap the existing components in a **`VotingKernel.process()`** pipeline. Keep submodules intact; the kernel imposes ordering, schema, and emits a single bundle.

**Pipeline per tick**
1. `committee = Committee.process()` → emits `proposal_vectors`, `member_confidences`, `committee_consensus`, `trade_vote` (legacy), `decision_id`  
2. `cons = ConsensusDetector.process(...)` → `consensus_score`  
3. `col = CollusionAuditor.process(...)` → `collusion_score`, `suspicious_pairs`  
4. `tha = TimeHorizonAligner.apply(...)` → `aligned_weights`, `horizon_alignment`  
5. `ars = AlternativeRealitySampler.process(...)` → `uncertainty.level` (+ compute **fragility**)  
6. `arb = StrategyArbiter.propose(...)` → `signals` and final **gate**  
7. Kernel publishes **`voting/decision_bundle`** and the canonical single‑writer keys.

**Why this helps**
- Deterministic ordering, one place to validate & normalize data shapes.
- One JSON to inspect per decision; **dramatically simpler debugging**.

---

## 6) Reliability & intelligence upgrades (path to 10/10)

1) **Per‑expert online reliability**  
   - Track Brier or LogLoss with exponential decay by regime/session.  
   - Weight = base_weight × reliability × regime‑match.  
   - Publish `expert_reliability` into the bundle for transparency.

2) **Uncertainty‑aware sizing**  
   - Compute **fragility** (fraction of alt worlds where action flips) from ARS.  
   - `risk_throttle = (1 - fragility) * (1 - collusion_score^α)`; clamp size by this.

3) **Collusion de‑bias**  
   - From suspicious pairs, derive **pair‑level penalties**; down‑weight consistently co‑moving experts beyond chance.

4) **Confidence calibration**  
   - Maintain calibration curves `p_hat → realized_win_prob` per regime. Rescale committee confidence before it reaches the Arbiter.

5) **Horizon intent loop**  
   - Committee publishes `intended_horizon_minutes`; THA aligns accordingly and returns an `impact_score` used by Arbiter.

6) **Schema locks & contracts**  
   - Enforce Voting Schema v1 in the kernel with hard **type/shape checks** and fail‑fast logs.

**Intelligence rating**  
- **Now:** 6.5/10 (solid analytics + safety nets; schema conflicts & missing learning loops).  
- **After upgrades:** 9–10/10 (coherent schema, learning, risk‑aware sizing, deterministic pipeline).

---

## 7) Robust debugging: **VotingDebugger** timeline

**What it records (per `decision_id`)**
- Stage events: inputs hash, outputs hash, success/fail, duration_ms
- Shape checks & diffs (e.g., `len(member_confidences)` vs `len(proposal_vectors)`)
- Key ownership conflicts (e.g., attempted write to `voting_consensus` by non‑owner)
- Gate decision with criteria breakdown
- File path + module version hints

**Where it stores**
- Files: `logs/voting/decision_<decision_id>.json`  
- Bus snapshot: `voting/debug/last_decision` (compact)  

**Example event entry**
```json
{
  "stage": "ConsensusDetector",
  "decision_id": "2025-09-18T19:05:33.421Z#128",
  "inputs_hash": "b3c1…",
  "outputs_hash": "9a44…",
  "duration_ms": 23,
  "checks": {"proposal_vectors": "ok", "member_confidences": "ok", "n_members": 5},
  "status": "ok"
}
```

---

## 8) Migration checklist (week‑by‑week)

**Week 1 (Schema & contracts)**
- Add `proposal_vectors`, `member_confidences`, `committee_members`, `decision_id` to committee output.
- Deprecate `voting_consensus` in committee; rename its dict to `committee_consensus`.
- Keep `ConsensusDetector` writing only `consensus_score` (remove its `voting_consensus` write after compatibility period).

**Week 2 (Kernel & Debugger)**
- Implement `VotingKernel.process()` with ordering + schema validation.  
- Add `VotingDebugger` and per‑tick JSON timelines.

**Week 3 (Learning & de‑bias)**
- Add per‑expert reliability; feed into weights.  
- Add `pair_penalties` to CollusionAuditor → committee weighting.  
- Use ARS to compute **fragility** and derive `risk_throttle` → Arbiter.

**Week 4 (Calibration & tests)**
- Implement confidence calibration by regime/session.  
- Build pytest suite with synthetic ticks to assert schema/owners/shapes.  
- Lock metrics dashboards (pass rate, consensus vs PnL, fragility vs drawdown).

---

## 9) Minimal bus contract (post‑migration)

- `voting/decision_bundle` (full JSON as in §4)  
- `trade_vote_v2` (flattened; single writer)  
- `consensus_score` (single writer: ConsensusDetector)  
- `collusion_score` (single writer: CollusionAuditor)  
- `aligned_weights` (single writer: THA)  
- `signals` (single writer: StrategyArbiter)

---

## 10) FAQ / Design rationale

- **Why both `committee_consensus` and `consensus_score`?**  
  Structured vs numeric; different consumers need each, avoiding type conflicts.

- **Why a single kernel if modules already exist?**  
  Deterministic ordering and one JSON bundle simplify debugging and prevent state mixing.

- **Do I have to give up namespaced diagnostics?**  
  No—keep rich, namespaced logs. Kernel only standardizes the **canonical** contract.

- **Is `proposal_vectors` redundant with `committee_votes`?**  
  No—vectors are for analytics; votes (dicts) are for human/debug UIs.
