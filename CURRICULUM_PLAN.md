# Curriculum & Training Plan

**Status:** proposal · **Date:** 2026-08-02 · **Branch:** `recovery/gate-1-fail-loud`
**Scope:** `envs/curriculum/` (10,895 LOC), `train/train_prop_firm.py`, `train/controllers/`

---

## 1. The reframe

The current curriculum makes **the world easier** and then hardens it.
Stage 0 trades at a 0.01-point spread with zero commission, zero slippage,
zero latency and a 50% drawdown allowance. Stage 9 trades at the real data
spread with commission, slippage, latency, shocks and a 4.8% daily limit.

That is a coherent idea, and much of it is well built. But it has a cost that
is easy to miss: **for the first ~100k steps the agent is rewarded for
behaviour that is only profitable because trading is free.** Everything it
learns there about how often to trade has to be un-learned later.

There is a different way to stage training, and it is roughly how large models
are actually trained: **do not make the world easier — make the learner better
prepared before it faces the world.**

| | Make the world easier | Make the learner readier |
|---|---|---|
| Stage 1 | zero-cost market | learn representations (supervised) |
| Stage 2 | small costs | imitate a competent teacher |
| Stage 3 | real costs | RL, anchored to the teacher |
| Risk | teaches habits that must be undone | none of the above teaches a false market |

Both can coexist. The recommendation is to keep the parts of the current
curriculum that stage *constraints and data*, and replace the part that stages
*market realism* with a preparation pipeline.

---

## 2. What exists today — measured

A stage sets five independent axes. Both endpoints, read from
`envs/curriculum/curriculum_config.py`:

| Axis | Explorer (0) | Live Ready (9) |
|---|---|---|
| **Execution** | spread 0.01, commission 0, slippage 0, latency 0, no randomisation, `use_data_spread=False` | spread 0.22 **+ real data spread**, commission 3.0/lot, slippage σ 0.06, latency 1 bar, shocks p=1.5% ×2.5 |
| **Constraints** | 50 trades/day, DD 50%/50%, 20 consecutive losses, stop €1000, no session rules | 7 trades/day, DD **4.8%/8.5%**, 4 consecutive losses, stop €250, all session rules enforced |
| **Reward** | nearly all shaping disabled, `exploration_bonus=0.02` | full shaping active |
| **Competence** | win-rate 0.0, PF 0.0 (open) | win-rate and PF gates tightened |
| **Data difficulty** | volatility percentile 0–30%, trend clarity ≥ 0.5 | full range |

Smooth intermediate progressions (verified per stage):

```
min_win_rate       0.00  0.25  0.35  0.38  0.40  0.45  0.48  0.50  ...
min_profit_factor  0.00  0.30  0.70  0.80  0.85  1.00  1.10  1.18  ...
entry_quality_thr  0.10  0.12  0.20  0.30  0.35  0.42  0.47  0.50  ... 0.60
max_steps/episode  1500  1500  1800  1800  2000  2200  2400  2600  ...
```

The stage index also drives the controller schedules:

```
entropy bounds   (0.08, 0.25) -> (0.003, 0.015)
LR multipliers   (0.5,  2.0)  -> (0.1,   0.4)
clip range       (0.15, 0.40) -> (0.04,  0.12)
```

**Assessment: this is the best-designed component in the repository and the
least validated.** The five-axis separation is genuinely good architecture.
Nothing about it has ever been tested against an alternative.

---

## 3. Three concrete defects

### 3.1 The terminal stage trains on the wrong cost regime — highest priority

Live Ready uses `use_data_spread=True, data_spread_scale=1.0` — the historical
spread from the tracked CSVs.

```
XAUUSD M15 spread, 2021-09 → 2025-12  (training data):  median  8.0 points
XAUUSD M15 spread, 2025-12 → 2026-07  (current):        median 41.0 points
```

**A "live ready" graduate has never seen the cost structure it would trade in.**
Against a mean absolute M15 move of 1.55, a 41-point spread is ~26% of a
typical bar. The agent is being certified against a market roughly 5× cheaper
than the real one.

### 3.2 Dead configuration

Every stage sets `include_memory_features` and `include_world_model_features`.
Both blocks were removed in observation schema v7.0. The flags are inert today
but will mislead the next person to tune a stage.

### 3.3 Competence is measured in-sample

Promotion runs `_run_validation_gate_episodes` against `self.training_env` —
the same data the agent trains on. This is **not** holdout leakage (an earlier
claim of mine that was wrong: `curriculum_callback.py` contains no `holdout`
references). It is the weaker problem that a stage gate can be passed by
memorisation rather than skill.

### 3.4 Half the curriculum never requires profitability

`min_profit_factor` first reaches 1.0 at stage 5. Stages 0–4 promote a
strictly loss-making policy. Defensible as exploration, but it means five
stages of learning are graded on activity rather than edge.

---

## 4. What transfers from how large models are trained

Four ideas transfer cleanly. They are not analogies for their own sake —
each addresses a specific measured weakness here.

### 4.1 Representations before policy

A language model is not taught helpfulness from scratch by reinforcement
learning. It learns representations first, by self-supervised prediction over
far more data than any RL stage could provide, and only then learns behaviour.

This project asks PPO to learn **representation and policy simultaneously**
from ~3,223 effectively independent samples (99,908 bars ÷ 31-bar
autocorrelation decay). That is the single hardest thing being asked of it,
and it is being asked for free.

> **Proposal — Phase A.** Pretrain the policy's feature encoder on supervised
> auxiliary targets computed from the same 40-dim observation: next-bar
> direction, realised volatility over the next N bars, and whether the next
> N bars exceed a cost threshold. These labels are free — they come from data
> already held — and they are far denser than trading reward. Freeze or
> warm-start the encoder into PPO.

Auxiliary prediction also gives an honest early read: **if a supervised head
cannot predict direction out-of-sample, PPO will not either.** It is strictly
the easier problem on the same data.

### 4.2 Imitation before exploration

Instruction tuning precedes preference optimisation. The model is shown
competent behaviour before being asked to search for it.

Random exploration in a market is expensive and mostly noise. Stage 0 currently
buys exploration by removing costs — which distorts the world. Behaviour
cloning buys it by showing examples — which does not.

> **Proposal — Phase B.** Behaviour-clone a rule-based teacher (a session
> filter plus a trend rule is enough) under **real** costs, until the policy
> reproduces it. Then hand the weights to PPO. The agent begins in a sensible
> region of policy space having never been told trading is free.

### 4.3 Anchor the policy to a reference

RLHF constrains the policy with a KL penalty to the reference model. Without it
the policy drifts into whatever exploits the reward signal — the reward model
is imperfect, and optimisation finds its holes.

This project has **~14 active shaping groups** and a `max_shaping_to_pnl_ratio`
cap that is itself an admission the reward is gameable. None of the shaping is
potential-based, so all of it changes the optimal policy (Ng et al., 1999).

> **Proposal — Phase C.** Add a KL penalty to the Phase-B reference policy,
> annealed down as training proceeds. This is a principled, well-tested defence
> against reward hacking, and it costs one term in the loss.

### 4.4 Evaluate on data you never train on, and stop on it

Benchmarks are held out and never trained against; checkpoint selection is on
eval, not train loss.

The 2026 FTMO period (14,451 M15 bars: −5.5% net, 2.25× volatility, 5.1×
spread) is exactly that — a genuine out-of-regime test that already exists.

> **Proposal.** It is opened once, at the end. Never for promotion, never for
> checkpoint selection, never for hyperparameter choice.

### 4.5 What does *not* transfer

Scale. The lesson from large models is that scale substitutes for cleverness —
and there is no scale available here. 4.2 years of one instrument is the
budget, and it is fixed. Every idea above is about **extracting more signal per
sample**, not about training longer. Increasing timesteps on the same data adds
gradient steps, not information.

---

## 5. Proposed training pipeline

```
Phase A — Representation            supervised, ~30 min
  40-dim observation -> encoder -> {next-bar direction, next-N volatility,
                                    move-exceeds-cost}
  Gate: out-of-sample AUC > 0.53 on at least one head, stable across 3 seeds.
  FAIL => stop. No edge is visible in these features; RL will not find one.

Phase B — Imitation                 behaviour cloning, ~1 h
  Teacher: session filter + trend rule, evaluated under REAL costs.
  Student: PPO policy network, warm-started from Phase A encoder.
  Gate: student reproduces >85% of teacher actions on held-out bars.

Phase C — Reinforcement             PPO + KL-to-reference
  Reward: net PnL after costs, drawdown as a CONSTRAINT.
  KL anchor: Phase-B policy, coefficient annealed 0.1 -> 0.
  Curriculum: constraints and data difficulty only (see §6).
  Gate: beats the Phase-B teacher on validation, 5 seeds.

Phase D — Validation
  Regime holdout, cost stress at 1.5x and 2x, seed spread, block bootstrap.

Phase E — Final test
  2026 data. Opened once. Selection rules frozen beforehand.
```

Phase A is the cheapest decisive experiment in the entire program. It runs in
minutes and can invalidate everything downstream.

---

## 6. Revised curriculum design

Keep the five-axis architecture. Change what each axis is allowed to do.

| Axis | Decision | Rationale |
|---|---|---|
| **Execution** | **Stop annealing.** Real costs from step 0, at *current* spreads. | Free-cost training teaches a trading frequency that is only viable at 0.01-point spreads. Exploration comes from Phases A–B instead. |
| **Constraints** | **Keep the ramp.** 50% DD → 4.8% is sound. | Loosening a *risk limit* does not misrepresent the market. It lets the agent experience the consequence of a drawdown before being terminated for it. |
| **Data difficulty** | **Keep the ramp.** Low-volatility, high-trend-clarity first. | The closest analogue to a data-mixture curriculum, and the axis most likely to help. Already properly implemented — precomputed percentiles and weighted episode sampling. |
| **Reward** | **Flatten.** Net PnL minus costs throughout; drawdown as a constraint. | Staged shaping means the objective changes underneath the agent. Reward should be the one thing that stays fixed. |
| **Competence** | **Keep, but evaluate out-of-sample.** | Gates are well-graduated. The problem is the data they are measured on. |

**Net effect: 10 stages of 5 axes → 10 stages of 2 axes.** The controller
schedules (entropy, LR, clip) stay indexed by stage, so the annealing survives.

---

## 7. Ablation program

Because the axes are independent, leave-one-out isolates the contribution of
each. **300k steps, 3 seeds per arm** — comparing learning curves, not final
performance.

| Arm | Setup | Question |
|---|---|---|
| **A** Flat | Live Ready settings from step 0 | Is any curriculum needed? |
| **B** Full | current 10 stages, 5 axes | Baseline to beat |
| **C** No cost ramp | execution fixed at real, others anneal | **Does free-cost exploration help, or teach overtrading?** |
| **D** No constraint ramp | constraints fixed at final | Does loose DD early help? |
| **E** No reward ramp | final reward from step 0 | Does staged shaping matter? |
| **F** No data ramp | full volatility range from step 0 | Does easy-data-first help? |
| **G** 3 stages | easy / medium / hard on all axes | Is 10-stage granularity worth 10,895 lines? |

**Metric:** net return per unit MaxDD on validation, plus steps-to-competence.
**Decision rule:** any axis whose removal costs < 0.10 Sharpe is flattened to a
linear schedule.

**Stated prior, so it is falsifiable:** arm **C beats B**. Training at
0.01-point spreads for 100k+ steps teaches a trade frequency that later stages
must suppress, and the suppression costs more than the exploration bought.

---

## 8. Fix checklist

Ordered by value, independent of the ablation.

- [ ] **Terminal cost regime.** Raise `data_spread_scale` to match current
      spreads, or run the final stage on 2026 data. *An agent certified against
      8-point spreads is not certified.*
- [ ] **Remove `include_memory_features` / `include_world_model_features`**
      from all 10 stage configs — they reference blocks deleted in v7.0.
- [ ] **Promote on unseen data.** A rolling validation slice, distinct from
      training and from the final 2026 test.
- [ ] **Surface stage + axis values on the dashboard.** Currently the run's
      trajectory through the curriculum is only visible in logs.
- [ ] **Fixed default seed.** `--seed -1` is time-based, which undoes the
      determinism the simulation clock provides.
- [ ] **Retire the `min_minutes_*` constraint fields** now that
      `min_bars_between_entries` / `min_bars_after_loss` are authoritative.

---

## 9. Acceptance gates

| Gate | Criterion | On failure |
|---|---|---|
| A — Representation | OOS AUC > 0.53, 3 seeds | **Stop.** Redirect to signal sourcing. |
| B — Imitation | >85% teacher-action agreement | Fix the teacher or the encoder |
| C — RL | beats teacher on validation, 5 seeds | Reward or algorithm is wrong |
| D — Robustness | positive at 1.5× costs; seed IQR < median | Not deployable |
| E — Final | positive on 2026 regime data after costs | **Stop.** No edge in this venue. |

---

## 10. What would falsify this plan

State these now, so the goalposts cannot move later:

1. **Phase A fails** — no supervised head beats 0.53 AUC out-of-sample. Then
   the features carry no signal and no amount of RL machinery will help.
2. **Arm A wins the ablation** — flat training matches the full curriculum.
   Then 10,895 lines should be deleted, and the effort was in the wrong place.
3. **Buy-and-hold beats everything on 2026 after costs.** Then the venue, not
   the method, is the problem.

Any of the three is a good outcome in the sense that matters: it stops a
larger investment in a direction that will not pay.

---

## 11. Honest note on what this changes

This plan improves the **probability of learning correctly**. It does not
create an edge, and nothing in it should be read as implying one exists.

The measured constraint is unchanged: ~3,223 effectively independent samples,
one instrument, a single strongly directional training regime (+149.8%), and a
current cost structure 5× worse than the one in the training data. Phases A–C
extract more from that budget. They do not enlarge it.

The strongest argument for doing Phase A first is that it costs half an hour
and can end the project honestly.
