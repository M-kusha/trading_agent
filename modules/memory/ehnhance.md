SmartInfoBus — Memory System 5/5 Upgrade Plan

This doc captures what we have now and everything we need to change to turn memory from a passive historian into a behavior-changing, self-improving guardrail that prevents repeat losers, adapts sizing/limits, and justifies its calls.

1) What we have today (ground truth)

UnifiedMemory orchestration

Coordinates components (replay, mistakes, playbook, neural, compression, budget), merges their outputs, and writes fixed SmartInfoBus keys. Handles circuit-breaker, parallel/sequential execution, health/metrics.

Bus update list is static: replay/budget/compression/mistakes/neural/playbook keys only (no gate/vote yet).

Stores new experiences (requires trade.pnl), with context metadata {regime, volatility, session, episode}.

Mistake memory (loss/win clustering)

DBSCAN over separate loss/win streams; keeps danger_zones (loss) & profit_zones (win) with scaler isolation, plus quality tracking. Emits avoidance_signal and similarities.

Playbook (kNN recall)

Scales, fits KNN, returns expected_pnl, confidence, recommended_action, and recall stats; also runs storage of recent trades.

Neural / Compression / Replay

Neural: embedding buffer & attention retrieval; Compression: PCA-style intuition_vector; Replay: sequence analysis & learning metrics. (Merged/published by UM via mapping.)

Unified memory store

Cosine similarity matrix with zero-pad align; retention score = 0.4importance + 0.3recency + 0.2access + 0.1|pnl|; cleanup under pressure.

Feature extraction (shared)

Market features include regime (one-hot-ish), session, volatility, plus trade/observation chunks with fixed dim packing.

Gap summary: We produce signals (avoidance, expected_pnl, intuition) but do not publish a pre-trade, authoritative memory_gate (veto/size/SL/TP), and we don’t provide a calibrated memory_vote for the ensemble. Also lacking: counterfactual interventions, regime-sharded retrieval, and sign-aware retention.

2) What “5/5 intelligent” means (behavioral goals)

Guard: Block or downsize setups similar to past losers before voting.

Adjust: Propose data-backed multipliers (size/SL/TP) from learned interventions.

Explain: Ship compact evidence (pattern, regime, top matches, counterfactual delta).

Adapt: Learn per-regime, recalibrate loss-risk, retain scar patterns longer.

3) New public contracts (add to SmartInfoBus)
3.1 memory_gate — authoritative pre-trade control
{
  "veto": true,
  "size_mult": 0.45,
  "sl_mult": 1.15,
  "tp_mult": 0.95,
  "confidence": 0.82,
  "reasons": [
    {"type":"pattern","label":"TRND-COMPR-BRK","regime":"volatile","similarity":0.67,
     "stats":{"n":47,"winrate":0.36,"avg_pnl":-8.2}},
    {"type":"counterfactual","best_alt":"halve_size","exp_delta":+5.4}
  ]
}

3.2 memory_vote — deliberation input
{ "score": -0.58, "confidence": 0.74, "uncertainty": 0.18, "rationale_ref": "memrat_2025-09-18_001" }

3.3 memory_rationale/<id> — evidence bundle
{
  "pattern_label": "TRND-COMPR-BRK",
  "regime": "volatile",
  "danger_similarity": 0.67,
  "loss_prob": 0.61,
  "stats": {"n": 47, "winrate": 0.36, "avg_pnl": -8.2},
  "top_matches": [{"sim":0.83,"pnl":-12.1,"age_h":5.2}, {"sim":0.79,"pnl":-7.6,"age_h":11.8}],
  "counterfactual": {"best_alt":"halve_size","exp_delta":5.4}
}


These are new keys to be appended to UM’s bus update list (see current list).

4) File-by-file changes (surgical, in order)
4.1 modules/memory/unified_memory.py

Add

_compose_gate_and_vote(component_results, context) -> Dict[str, Any]

Inputs: mistakes.avoidance_signal, danger_similarity/profit_similarity; playbook.expected_pnl/confidence/recommended_action; (new) loss_risk_head.loss_prob/uncertainty; optional compression.intuition_vector.

Logic:

risk_score = max(loss_prob, danger_similarity)

veto = (risk_score > τ1) and (avoidance_signal > τ2)

size_mult = clamp(1 - k1*risk_score + k2*profit_similarity, 0.2, 1.5)

sl_mult/tp_mult from intervention table (below)

vote.score = tanh(expected_pnl / s) - k3*risk_score

vote.confidence = min(1, playbook.confidence * (1 - uncertainty))

Extend _update_all_bus_keys to publish:
memory_gate, memory_vote, and memory_rationale/<id> (keep thesis & logging). Current writer loops over a fixed list—append these keys.

In _store_experiences → enrich metadata with pattern_label (see Section 5.2). Currently stores regime/vol/session/episode.

Why: UM already merges/publishes; we only compose & add the new outputs in one place.

4.2 modules/memory/components/mistakes.py

Add

After _calculate_avoidance_signals, compute a compact gate snippet:

risk_multiplier = 1 + 1.5 * max(0, danger_similarity - profit_similarity)

veto = (danger_similarity > 0.55 and avoidance_signal > 0.6)

Return this as part of component result (non-breaking field), so UM can consume it.

Keep

Existing DBSCAN clustering for loss/win; similarities and avoidance_signal are correct and stable.

4.3 modules/memory/components/playbook.py

Add

In recall result:

signed_bias = tanh(expected_pnl / pnl_scale)

neighbors: top-K {dist, pnl} for rationale bundle; keep confidence.
Keep KNN fit/recall & analytics.

4.4 modules/memory/shared/memory_store.py

Change (sign-aware retention)

In _calculate_retention_score: add loss_bonus so big losers persist longer:

loss_bonus = max(0.0, -pnl)/50.0; score = 0.35*importance + 0.25*recency + 0.15*access + 0.05*|pnl| + 0.20*loss_bonus
Current weightings are sign-agnostic (|pnl| only).

4.5 modules/memory/shared/feature_extractor.py

Add (bar-signature dims)

Extend trade/market extraction with ~8–12 compact shape features: slope_10/30, atr_jump, compression_z, breakout_dist, wick_body_ratio, range_frac, small shape_code. Respect fixed total dim via existing _ensure_dim.

4.6 modules/memory/components/compression.py (optional but strong)

Use

intuition_vector.strength to slightly tilt memory_vote.score (+ if profit-aligned, − if loss-aligned). UM reads; no new public key required.

4.7 modules/memory/components/neural.py (light touch)

Expose

neural_risk_hint = 1 - max(attention_weights) as an uncertainty contributor for the vote. UM blends this into vote.uncertainty.

5) New components to add
5.1 modules/memory/components/loss_risk_head.py

Purpose: Online calibrated head: (features, action, regime) → P(loss > τ)

Output: {"loss_prob": p, "uncertainty": u, "calibration_ece": ece}

Integration: UM fuses into risk_score = max(loss_prob, danger_similarity), and weights memory_vote by (1 - uncertainty).

5.2 modules/memory/components/interventions.py

Purpose: Anti-relapse table learned from counterfactuals over nearest neighbors.

State:

{ (pattern_label, regime) -> {intervention, strength, best_alt, exp_delta, n} }


Output:
{"intervention":"avoid|halve_size|tighter_sl|…","strength":0..1,"best_alt":str,"exp_delta":float}

Integration: UM maps to size_mult/sl_mult/tp_mult; sets veto when intervention=='avoid' && strength>τ.

5.3 modules/memory/shared/bar_signature.py (if preferred)

Helper that transforms a short OHLCV window into the 8–12 signature features used by the extractor.

6) Interfaces with Voting/Position/Execution/Core

PositionManager / Execution

If memory_gate.veto == True → NO_TRADE.

Else apply size_mult/sl_mult/tp_mult to the plan before sending orders.

Log reasons[] onto the order annotations for audit.

Ensemble/Voting

Treat memory_vote like any voter: weight = confidence * (1 - uncertainty) * module_trust.

Keep memory as both: a gate (policy constraint) and a vote (preference).

Core / Bus contracts

Add memory_gate, memory_vote, and memory_rationale/* to SmartInfoBus schema & ingestion; extend UM set loop (current list is in _update_all_bus_keys).

7) Data & labels we must add on store

In UM _store_experiences, append to metadata:
pattern_label (from a discretizer/pattern detector), plus vol_bucket if available alongside existing regime/volatility/session. Current store already accepts arbitrary metadata.

8) KPIs & diagnostics to track weekly

Repeat-Loss Prevention Rate (RLPR): % of loss-patterns subsequently vetoed or downsized.

Counterfactual Gain: Σ expected_delta realized vs proposed.

Gate Precision/Recall: avoided losers vs blocked winners.

Calibration ECE (loss head): lower is better.

Zone Drift: silhouette ↓ or center shift triggers recluster.

UM’s thesis builder already aggregates component snippets; append these KPIs there.

9) Test plan (no-regret rollout)

Unit

Mistakes: danger/profit similarities monotonicity w.r.t. center distance.

Playbook: recall conf increases as neighbor distances shrink; signed_bias ∈ [−1,1].

Store: retention preference for negative pnl; pruning respects pressure.

Integration (shadow mode)

Publish memory_gate/memory_vote but don’t enforce veto for 1–2 days; record what would have been blocked and delta PnL.

Live

Enable veto with conservative thresholds; monitor RLPR and Gate Precision.

Gradually enable size_mult/sl_mult/tp_mult from interventions.

10) Risks & mitigations

Over-blocking: Start with soft gate (size_mult only) + high thresholds; promote to veto after good precision.

Regime shift: Shard by regime/vol bucket (retrieve/cluster within shard); drift alarms trigger recluster.

Data leakage: Keep loss/win scalers separate (already done).

11) Implementation checklist (copy/paste)

 unified_memory.py: add _compose_gate_and_vote, enrich _store_experiences, extend _update_all_bus_keys with new keys.

 mistakes.py: return compact gate snippet (risk_multiplier, veto, confidence, reasons) alongside existing outputs.

 playbook.py: add signed_bias and top-K neighbors in recall output.

 memory_store.py: sign-aware retention adjustment.

 feature_extractor.py (and/or bar_signature.py): add bar-signature dims; keep total dim stable.

 Add new components: loss_risk_head.py, interventions.py; wire into UM compose.

 Core/Bus: register new keys in schema; PositionManager reads & enforces memory_gate.

 Ensemble: include memory_vote with calibrated weight.

 KPIs: compute & append to UM thesis.