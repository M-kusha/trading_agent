

# 📒 Master ledger (I’ll maintain this as we go)

For each module I’ll append a row:

* `Module` · `File` · `Category` · `Provides` · `Requires` · `Optional Inputs` · `Outputs (shape)` · `Side-effects (bus/persist/logging)` · `Voting?` · `Explainable?` · `Timeout/Priority` · `Key Methods` · `State` · `Error Paths` · **Consumers of its outputs** (filled once other modules arrive) · **Owners of its inputs** (filled once discovered) · **Conflicts/Duplicates** (live)

---

# RAW CAPTURE — AuditingCoordinator

**File:** `modules/auditing/auditing_coordinator.py`
**Class:** `AuditingCoordinator(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `provides = ['audit_status', 'audit_report', 'audit_metrics']`
* `requires = ['trading_signal', 'market_data', 'trades']`
* `category = 'auditing'`
* `is_voting_member = False`
* `explainable = True`
* `hot_reload = True`
* `timeout_ms = 150`
* `priority = 3`
* `version = "2.0.0"`

## 1) Public API

* `def _initialize(self) -> None`
* `def _discover_audit_modules(self) -> None`
* `def reset(self) -> None`
* `async def process(self, **inputs) -> Dict[str, Any]`
* `def _extract_audit_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]`
* `async def _perform_comprehensive_audit(self, audit_data: Dict[str, Any]) -> Dict[str, Any]`
* `def _audit_trading_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]`
* `def _audit_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]`
* `def _determine_overall_audit_status(self, results: Dict[str, Any]) -> str`
* `def _validate_cross_module_consistency(self, audit_results: Dict[str, Any]) -> Dict[str, Any]`
* `def _generate_unified_audit_report(self, audit_results: Dict[str, Any], validation_results: Dict[str, Any]) -> str`
* `def _calculate_audit_metrics(self, audit_results: Dict[str, Any]) -> Dict[str, Any]`
* `def _generate_audit_thesis(self, audit_results: Dict[str, Any], validation_results: Dict[str, Any]) -> str`
* `def _update_audit_performance(self, success: bool, audit_time: float) -> None`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`  *(exposes an action spec but module is not a voting member)*
* `async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float`
* `def get_comprehensive_status(self) -> Dict[str, Any]`

## 2) Inputs (from `requires` and code usage)

Expected at `process(**inputs)` or derived inside:

* `trading_signal: dict`
  used keys (if present): `confidence: float`, `reason: str` or `_thesis: str`
* `market_data: dict`
  not read in current logic (just passed through to audit data)
* `trades: list[dict]`
  used key per trade: `pnl` (numeric)
* optional extras the code handles:

  * `timestamp: datetime` (defaults to `datetime.now()` if missing)
  * `step_idx: int` (defaults to `self._step_count` if missing)

## 3) Outputs (from `provides` and return)

Returned dict from `process()`:

```python
{
  'audit_status': str,                        # 'excellent' | 'good' | 'fair' | 'poor' | 'unknown' | 'failed'
  'audit_report': str,                        # multiline human-readable report
  'audit_metrics': {                          # dict with:
     'signal_quality': float,                 # 0..1
     'trade_quality': float,                  # 0..1
     'confidence': float,                     # (signal_quality + trade_quality)/2
     'audit_count': int,                      # session total
     'success_rate': float                    # 0..1 over session
  },
  '_thesis': str                              # explainability payload (because explainable=True)
}
```

## 4) Side-effects / Bus / Logging

* Calls `InfoBusManager.get_instance()` and **writes directly** to bus once:

  * `smart_bus.set('audit_status', <status>, module=<class name>, thesis=<str>, confidence=<float>)`
* Also returns all three provided keys; orchestrator will typically publish those too (outside this file).
* Logging via `self.logger.info/error` with status and counts.

## 5) Internal state & lifecycle

* State set in `_initialize()`:

  * `discovered_auditors: dict` (currently reset to `{}` by `_discover_audit_modules()`)
  * `audit_modules: list` (unused in current snippet)
  * `cross_validation_cache: dict`
  * `audit_session_start: datetime`
  * `audit_performance: dict` with `total_audits`, `successful_audits`, `failed_audits`, `avg_audit_time`
* `reset()` clears caches, resets timers/counters, and logs.

## 6) Core logic (summaries)

* `_extract_audit_data` → packs `trading_signals`, `market_data`, `trades`, `timestamp`, `step_idx`.
* `_perform_comprehensive_audit`:

  * initializes `results = {'trade_audit':{}, 'thesis_audit':{}, 'signal_audit':{}, 'overall_status':'healthy'}`
  * if signals exist → `_audit_trading_signals`
  * if trades exist → `_audit_trades`
  * then `_determine_overall_audit_status`
  * on exception → sets `overall_status='failed'` and attaches `'error'`.
* `_audit_trading_signals`:

  * `signal_count = 1 if signals else 0`
  * flags: `has_confidence`, `has_thesis`
  * `quality_score` adds: +0.3 if has\_confidence, +0.4 if has\_thesis, +0.3 if `confidence>0.5`
* `_audit_trades`:

  * if empty → `{'trade_count':0, 'quality_score':1.0}`
  * else → `total_pnl = sum(pnl)`, `win_rate = profitable/len(trades)`, `quality_score = min(1.0, win_rate+0.3)`
* `_determine_overall_audit_status`:

  * averages any available `quality_score` values across sub-results
  * thresholds: `>=0.8: excellent`, `>=0.6: good`, `>=0.4: fair`, else `poor`
* `_validate_cross_module_consistency`:

  * compares `signal_count` vs `trade_count`
  * if `abs(diff) > 5` → append conflict and `consistency_score *= 0.8`
* `_generate_unified_audit_report`:

  * builds a multi-line report with timestamp, session duration (hours), overall status,
    % scores, perf stats, and lists conflicts if present.
* `_calculate_audit_metrics`:

  * `overall_confidence = (signal_quality + trade_quality)/2`
  * includes session totals and success rate.
* `_generate_audit_thesis`:

  * positive/neutral/attention-needed message based on `status`, `consistency_score`, and conflict count.
* `_update_audit_performance`:

  * increments counters; updates `avg_audit_time` via EMA (`alpha=0.1`).

## 7) Concurrency / timing

* `process` and two helper methods are `async`.
* Metadata has `timeout_ms=150` (enforced outside this file).

## 8) Error paths & fallbacks

* Any exception in `process`:

  * logs `[FAIL]`
  * updates performance with `success=False`
  * returns safe result with `audit_status='failed'`, `confidence=0.0`, `_thesis` describing the error.
* Any exception in `_perform_comprehensive_audit`:

  * `overall_status='failed'` and an `error` field in results.

## 9) Voting / explainability flags

* `is_voting_member = False`
* Provides two hooks anyway:

  * `propose_action` → describes an `audit_coordination` action with target modules = keys of `discovered_auditors`.
  * `calculate_confidence` → returns confidence based on session success rate (capped to 0.9).
* `explainable = True` → `_thesis` is included in returns and also attached to bus set for `audit_status`.

## 10) Connections (to be filled as we ingest other files)

* **Consumes:** `trading_signal`, `market_data`, `trades`
  *(owners TBD — will be filled from other module captures and/or Contracts file)*
* **Provides:** `audit_status`, `audit_report`, `audit_metrics`
  *(consumers TBD — monitoring/UX modules likely; will fill when we see them)*

## 11) Gaps/notes (no action now)

* `_discover_audit_modules()` currently does not populate `discovered_auditors`; discovery mechanism likely handled by orchestrator or pending.
* `market_data` is required but not used inside audits (kept for future consistency checks).
* Only `audit_status` is written directly to the bus; `audit_report`/`audit_metrics` are returned but not explicitly `set()` here (likely set by orchestrator).


## 📒 Master ledger (append)

**Module:** TradeExplanationAuditor
**File:** `modules/auditing/trade_explanation_auditor.py`
**Category:** auditing
**Provides:** `trade_explanations`, `audit_alerts`, `explanation_metrics`
**Requires:** `trading_signal`, `market_data`, `trades`
**Optional Inputs (observed):** `timestamp`, `step_idx`
**Side-effects:** writes `trade_explanations` to SmartInfoBus; logs operator messages; keeps deque state
**Voting?** no (`is_voting_member=False`)
**Explainable?** yes (`explainable=True`)
**Timeout/Priority:** `timeout_ms=100`, `priority=3`
**Key Methods:** `process`, `_audit_single_trade_explanation`, `_validate_explanation_quality`, `_analyze_explanations`, `_check_for_alerts`, `_update_quality_metrics`, `generate_detailed_report`, `get_explanation_statistics`
**State:** deque `trade_explanations`, counters in `quality_metrics`, `session_start`, `explanation_cache`, thresholds in `alert_thresholds`
**Error paths:** try/except in `process` and per-trade auditing
**Consumers (TBD):** (filled as other modules arrive)
**Owners of inputs (TBD):** (filled as other modules arrive)
**Conflicts/Duplicates:** none observed in this file

---

# RAW CAPTURE — TradeExplanationAuditor

**Class:** `TradeExplanationAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin)`
**Decorator metadata (@module):**

* `provides = ['trade_explanations', 'audit_alerts', 'explanation_metrics']`
* `requires = ['trading_signal', 'market_data', 'trades']`
* `category = 'auditing'`
* `is_voting_member = False`
* `explainable = True`
* `hot_reload = True`
* `timeout_ms = 100`
* `priority = 3`
* `version = "2.0.0"`

## 1) Public API (methods)

* `def _initialize(self) -> None`
* `def reset(self) -> None`
* `async def process(self, **inputs) -> Dict[str, Any]`
* `def _extract_audit_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]`
* `def _process_trade_explanations(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]`
* `def _audit_single_trade_explanation(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]`
* `def _validate_explanation_quality(self, trade_explanation: Dict[str, Any], trade: Dict[str, Any]) -> None`
* `def _analyze_explanations(self, processed_trades: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]`
* `def _check_for_alerts(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]`
* `def _update_quality_metrics(self, analysis: Dict[str, Any]) -> None`
* `def _generate_explanation_thesis(self, analysis: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str`
* `def generate_detailed_report(self) -> str`
* `def get_explanation_statistics(self) -> Dict[str, Any]`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`
* `async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float`

## 2) Inputs (from `requires` + code usage)

Expected via `process(**inputs)` or derived:

* `trading_signal: dict` (passed into context; not directly scored in quality)
* `market_data: dict` (passed into context; not directly used in scoring)
* `trades: list[dict]` — iterated; each trade can include:

  * `symbol: str`
  * `timestamp: any` (used to compose `trade_id`)
  * `action: str` (quality boost if `'buy'|'sell'|'hold'`)
  * `pnl: number`
  * `confidence: float` (affects quality)
  * `reason: str` or `_thesis: str` (counts as “has thesis”)
  * optional: `risk_assessment` (presence adds to quality)
* Optional extras handled:

  * `timestamp: datetime` (defaults to `datetime.now()`)
  * `step_idx: int` (defaults to `0`)
* Internal context adds: `risk_score: 0.5` (constant default in this file)

## 3) Outputs (from `provides` + return)

`process()` returns:

```python
{
  'trade_explanations': dict,      # analysis dict (see §5)
  'audit_alerts': list[dict],      # alerts based on rates/thresholds
  'explanation_metrics': dict,     # running session metrics (see §6)
  '_thesis': str                   # explainability payload
}
```

Additionally, the module **writes directly** to the bus:

* `smart_bus.set('trade_explanations', explanation_analysis, module=<class>, thesis=<str>, confidence=<float>)`

## 4) Side-effects / Logging

* SmartInfoBus write: **only** `trade_explanations` is set from inside the module.
* Operator logs via `self.logger.info` using `format_operator_message(...)` when:

  * `abs(pnl) > 50` **or** `explanation_quality < 0.5`.
* Errors logged with `[FAIL]` prefixes.

## 5) Core logic & scoring

### 5.1 Per-trade explanation object (from `_audit_single_trade_explanation`)

Fields included:

* `trade_id = f"{symbol}_{timestamp}"`
* `symbol`, `action` (default `'unknown'`), `pnl` (default `0`)
* `confidence` (default `trade.get('confidence', 0.5)`)
* `has_thesis` = `bool(trade.get('reason') or trade.get('_thesis'))`
* `market_regime` = `context.get('market_regime', 'unknown')` *(note: `market_regime` not populated in this file)*
* `risk_level` = `'high'` if `context['risk_score'] > 0.7` else `'normal'`
* `processed_at` = ISO timestamp
* `explanation_quality` (computed)
* `quality_issues` (list of strings)

### 5.2 Quality scoring (from `_validate_explanation_quality`)

* Start at `0.0`
* **Thesis present** (`has_thesis`): `+0.4`; else add issue `missing_thesis`
* **Confidence:**

  * `> 0.7` → `+0.3`
  * `< 0.3` → add issue `low_confidence` and `+0.1`
  * else → `+0.2`
* **Action clarity:** action in `{'buy','sell','hold'}` → `+0.2`; else add `unclear_action`
* **Risk assessment:** if `'risk_assessment' in trade` → `+0.1`; else add `missing_risk_assessment`
* Max theoretical score: **1.0**

### 5.3 Batch analysis (from `_analyze_explanations`)

If no processed trades:

* returns `trade_count=0`, `avg_quality=1.0`, `avg_confidence=1.0`, `overall_confidence=1.0`, `pattern_analysis="No trades to analyze"`

Else computes:

* `avg_quality` = mean of `explanation_quality`
* `avg_confidence` = mean of `confidence`
* `missing_thesis_count` = count of `not t['has_thesis']`
* `low_confidence_count` = count of `t['confidence'] < 0.5`
* `overall_confidence` = `min(avg_quality, avg_confidence)`
* `pattern_analysis` string with summary
* `detailed_trades` = last 10 processed trades

### 5.4 Alerts (from `_check_for_alerts`)

If `trade_count == 0` → `[]`. Otherwise potential alerts:

* **Low explanation quality** if `avg_quality < 0.5`
  → `{type: 'low_explanation_quality', severity: 'high', ...}`
* **Low confidence pattern** if `avg_confidence < 0.4`
  → `{type: 'low_confidence_pattern', severity: 'medium', ...}`
* **Missing explanations** if `(missing_thesis_count / trade_count) > alert_thresholds['missing_explanation_rate']` (default `0.1`)
  → `{type: 'missing_explanations', severity: 'medium', ...}`
* **Risk-based** if `context['risk_score'] > 0.8` **and** `avg_confidence < 0.6`
  → `{type: 'high_risk_low_confidence', severity: 'critical', ...}`

### 5.5 Thesis (from `_generate_explanation_thesis`)

* If no trades: `"No trades processed for explanation auditing in this cycle."`
* Else:

  * `quality > 0.8` and `confidence > 0.7` → “Excellent …”
  * `quality > 0.6` and `confidence > 0.5` → “Good …”
  * otherwise → “Needs improvement …”
  * appends alert count if any.

## 6) Running session metrics (from `quality_metrics` + updater)

* `total_trades_audited`
* `high_confidence_trades` (+= `trade_count - low_confidence_count`)
* `low_confidence_trades`
* `missing_explanations`
* `pattern_violations`
  (+= number of trades in `detailed_trades` with `len(quality_issues) > 2`)

## 7) Reports & statistics

* `generate_detailed_report()` returns a multi-line string with:

  * session duration hours (since `session_start`)
  * totals and rates:

    * High Confidence Rate = `high_confidence_trades / total_audited`
    * Missing Explanations = `missing_explanations / total_audited`
    * Pattern Violations = `pattern_violations / total_audited`
  * recommendations text
* `get_explanation_statistics()` returns:

  * `total_explanations` (len deque)
  * `recent_explanations` (last 100)
  * `avg_quality`, `avg_confidence` over recent
  * `quality_metrics`
  * `session_duration_hours`

## 8) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Module metadata `timeout_ms=100` (enforced externally).

## 9) Voting / explainability

* Not a voting member.
* Provides a neutral `propose_action` with `_thesis` describing focus.
* `calculate_confidence`: if `total_trades_audited == 0` → `0.5`; else
  returns `min(0.9, max(0.1, high_confidence_rate * (1 - missing_explanation_rate)))`.

## 10) State & lifecycle

* `_initialize`:

  * `trade_explanations = deque(maxlen=1000)`
  * `session_start = now`
  * `explanation_cache = {}`
  * `quality_metrics` dict (see §6)
  * `alert_thresholds` dict:

    * `low_confidence_rate = 0.3` *(note: not directly used in alerts)*
    * `missing_explanation_rate = 0.1`
    * `pattern_violation_rate = 0.15` *(not directly used in alerts)*
* `reset` clears deque, resets counters and cache.

## 11) Data shapes (canonical, copy-paste)

```yaml
requires:
  trading_signal: dict
  market_data: dict
  trades: list[dict]
    - symbol: str
      timestamp: any
      action: str            # 'buy'|'sell'|'hold' preferred
      pnl: number
      confidence: float
      reason: str            # optional
      _thesis: str           # optional
      risk_assessment: any   # optional

provides:
  trade_explanations:
    trade_count: int
    avg_quality: float       # 0..1
    avg_confidence: float    # 0..1
    overall_confidence: float
    missing_thesis_count: int
    low_confidence_count: int
    pattern_analysis: str
    detailed_trades: list[dict]   # up to 10
  audit_alerts:
    - type: str
      severity: str          # 'high'|'medium'|'critical'
      message: str
      ...                    # may include rates/timestamp
  explanation_metrics:
    total_trades_audited: int
    high_confidence_trades: int
    low_confidence_trades: int
    missing_explanations: int
    pattern_violations: int
```

## 12) Minimal example I/O

```python
inputs = {
  "trading_signal": {"side":"BUY","confidence":0.62,"reason":"Breakout"},
  "market_data": {"symbol":"XAUUSD","price":2350.1},
  "trades": [
    {"symbol":"XAUUSD","timestamp":1724500000,"action":"buy","pnl":42.0,"confidence":0.8,"reason":"Momentum"},
    {"symbol":"XAUUSD","timestamp":1724500300,"action":"sell","pnl":-15.5,"confidence":0.4}
  ]
}

result = {
  "trade_explanations": {...},   # analysis dict per §5.3
  "audit_alerts": [...],         # possibly empty
  "explanation_metrics": {...},  # updated session counters
  "_thesis": "Good/Excellent/Needs improvement ... (+alerts if any)"
}
```

## 13) Connections (to fill as we ingest more files)

* **Consumes:** `trading_signal`, `market_data`, `trades` (owners TBD)
* **Provides:** `trade_explanations`, `audit_alerts`, `explanation_metrics` (consumers TBD)

## 14) Notes / gaps (no action now)

* `_extract_audit_context` sets a fixed `risk_score=0.5` (not sourced from bus).
* `market_regime` used in per-trade object comes from `context` but isn’t populated in this file.
* `alert_thresholds['low_confidence_rate']` and `['pattern_violation_rate']` exist but are **not** used directly in `_check_for_alerts`.

## 📒 Master ledger (append)

**Module:** TradeThesisTracker
**File:** `modules/auditing/trade_thesis_tracker.py`
**Category:** auditing
**Provides:** `thesis_analysis`, `thesis_performance`, `thesis_alerts`
**Requires:** `trading_signal`, `market_data`, `trades`
**Optional Inputs (observed):** `timestamp`, `step_idx`
**Side-effects:** writes `thesis_analysis` to SmartInfoBus; logs thesis changes; maintains transition/history state
**Voting?** no (`is_voting_member=False`)
**Explainable?** yes (`explainable=True`)
**Timeout/Priority:** `timeout_ms=100`, `priority=3`
**Key Methods:** `process`, `_extract_current_thesis`, `_handle_thesis_change`, `_update_thesis_performance`, `_generate_thesis_analysis`, `_check_for_thesis_alerts`, `get_comprehensive_thesis_analysis`, `generate_thesis_report`
**State:** `current_thesis`, `thesis_changes`, `thesis_performance` (defaultdict), `thesis_history` (deque), `thesis_patterns`, `transition_matrix`, `session_start`, `best_thesis`, `worst_thesis`, `alert_thresholds`
**Error paths:** try/except in `process`; per-trade processing guarded
**Consumers (TBD):** (fill when others arrive)
**Owners of inputs (TBD):** (fill when others arrive)
**Conflicts/Duplicates:** none observed in this file

---

# RAW CAPTURE — TradeThesisTracker

**Class:** `TradeThesisTracker(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `provides = ['thesis_analysis', 'thesis_performance', 'thesis_alerts']`
* `requires = ['trading_signal', 'market_data', 'trades']`
* `category = 'auditing'`
* `is_voting_member = False`
* `explainable = True`
* `hot_reload = True`
* `timeout_ms = 100`
* `priority = 3`
* `version = "2.0.0"`

## 1) Public API (methods)

* `def _initialize(self) -> None`
* `def reset(self) -> None`
* `async def process(self, **inputs) -> Dict[str, Any]`
* `def _extract_thesis_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]`
* `def _extract_current_thesis(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> str`
* `def _handle_potential_thesis_change(self, new_thesis: str) -> bool`
* `def _handle_thesis_change(self, old_thesis: str, new_thesis: str) -> None`
* `def _process_trades_with_thesis(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]`
* `def _process_single_trade_with_thesis(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]`
* `def _update_thesis_performance(self, processed_trades: List[Dict[str, Any]]) -> None`
* `def _update_best_worst_thesis(self) -> None`
* `def _generate_thesis_analysis(self) -> Dict[str, Any]`
* `def _check_for_thesis_alerts(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]`
* `def _generate_thesis_explanation(self, analysis: Dict[str, Any], thesis_changed: bool) -> str`
* `def get_comprehensive_thesis_analysis(self) -> Dict[str, Any]`
* `def generate_thesis_report(self) -> str`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`
* `async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float`

## 2) Inputs (from `requires` + code usage)

* `trading_signal: dict`

  * used via context as:

    * `signal_confidence = trading_signal.get('confidence', 0.5)`
    * `signal_action = trading_signal.get('action', 'unknown')`
    * `signal_reason = trading_signal.get('reason', '')`
* `market_data: dict`

  * optional usage in `_extract_current_thesis`:

    * expects `close: list[number]`; if len≥5, sets `market_regime` to `trending_up` if `close[-1]>close[-5]` else `trending_down`
* `trades: list[dict]`

  * consumed in `_process_trades_with_thesis`; each trade may include `symbol`, `timestamp`, `action`, `pnl`
* Optional extras handled:

  * `timestamp: datetime` (defaults to `datetime.now()`)
  * `step_idx: int` (defaults to `0`)

## 3) Outputs (from `provides` + return)

`process()` returns:

```python
{
  'thesis_analysis': dict,        # see §5.3
  'thesis_performance': dict,     # dict(view) of defaultdict -> {thesis: {trades,pnl,confidence}}
  'thesis_alerts': list[dict],    # alerts from §5.4
  '_thesis': str                  # explanation string
}
```

Additionally, the module writes directly to the bus:

* `smart_bus.set('thesis_analysis', thesis_analysis, module=<class>, thesis=<str>, confidence=<float>)`

## 4) Side-effects / Logging

* SmartInfoBus write: `thesis_analysis` (with thesis & confidence).
* Logs thesis changes via `format_operator_message("🧠", "Thesis change: old → new", ...)`.

## 5) Core logic

### 5.1 Thesis extraction (`_extract_current_thesis`)

* If no `trading_signal` → returns `"no_signal"`.
* Builds `confidence_level`: `high` if `signal_confidence>0.7`, `medium` if `>0.4`, else `low`.
* Computes `market_regime` from `market_data.close` (if available) as `trending_up`/`trending_down`; defaults to `unknown`.
* Base thesis string: `f"{action}_{market_regime}_{confidence_level}"`.
* If `signal_reason` contains any of `{bullish,bearish,breakout,reversal,momentum,support,resistance}` (case-insensitive), appends first found word to thesis.

### 5.2 Change handling (`_handle_potential_thesis_change` / `_handle_thesis_change`)

* Increments `thesis_changes` when `new_thesis != current_thesis`.
* Records transition in `transition_matrix[old][new] += 1`.
* Appends change record to `thesis_history` with timestamp and change number.
* Updates `current_thesis` and logs an operator message.

### 5.3 Trade processing & performance

* `_process_trades_with_thesis` builds `processed_trade` objects:

  * fields: `trade_id = f"{symbol}_{timestamp}"`, `symbol`, `action` (default `'unknown'`), `pnl` (default `0`), `thesis=current_thesis`, `confidence=context.signal_confidence`, `timestamp=context.timestamp`, `processed_at=ISO`.
* `_update_thesis_performance` updates:

  * per-thesis `trades += 1`, `pnl += trade.pnl`
  * per-thesis `confidence` EMA with `alpha=0.2` (seeded by first value)
* `_update_best_worst_thesis` sets `best_thesis`/`worst_thesis` by max/min total `pnl`.

### 5.4 Analysis & alerts

* `_generate_thesis_analysis` returns:

  * `current_thesis`, `thesis_changes`, `change_frequency` (changes/hour since session start), `active_thesis_count`,
  * `current_confidence`, `current_performance` (for current thesis),
  * `best_thesis`, `worst_thesis`,
  * `total_pnl`, `total_trades`,
  * `session_duration_hours`,
  * `thesis_diversity = active_thesis_count / max(thesis_changes, 1)`
* `_check_for_thesis_alerts` triggers:

  * **frequent\_thesis\_changes** (`severity='medium'`) if `change_frequency > alert_thresholds['frequent_changes']` (default 10/hour)
  * **poor\_thesis\_performance** (`severity='high'`) if `current_performance.pnl < alert_thresholds['poor_performance']` (default -100)
  * **low\_thesis\_confidence** (`severity='medium'`) if `current_confidence < alert_thresholds['low_confidence']` (default 0.3)
* `_generate_thesis_explanation` builds human-readable thesis status, performance note (positive/negative/breakeven), and diversity insight (low `<3` vs high `>5` active theses).

## 6) Running state & thresholds

* State initialized in `_initialize`:

  * `current_thesis="unknown"`, `thesis_changes=0`
  * `thesis_performance = defaultdict(lambda: {"trades":0,"pnl":0.0,"confidence":0.0})`
  * `thesis_history = deque(maxlen=500)`
  * `thesis_patterns = defaultdict(int)` *(not incremented in this file)*
  * `transition_matrix = defaultdict(lambda: defaultdict(int))`
  * `session_start = now`, `best_thesis=None`, `worst_thesis=None`
  * `alert_thresholds = {'frequent_changes':10,'poor_performance':-100,'low_confidence':0.3}`

## 7) Reports & statistics

* `get_comprehensive_thesis_analysis()` returns base analysis plus:

  * `transitions` (dict-of-dicts), `patterns` (dict copy), `recent_history` (last 20 changes),
  * `trading_summary` via `_get_trading_summary()` if available on mixin.
* `generate_thesis_report()` prints a formatted report including:

  * session duration, current thesis, change count,
  * current P\&L/confidence, total P\&L/trades,
  * best/worst thesis with P\&L,
  * change frequency, diversity, active thesis count.

## 8) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Metadata `timeout_ms=100` (enforced externally).

## 9) Voting / explainability

* Not a voting member.
* `propose_action` returns `{'action_type':'thesis_tracking', ... , '_thesis': '...'}`.
* `calculate_confidence` combines:

  * `stability_score = 1 - (thesis_changes / max(1, total_theses*10))` clamped `[0.1,1]`
  * `performance_score = (total_pnl + 1000)/2000` clamped `[0.1,1]`
  * final `confidence = (stability_score + performance_score)/2` clamped `[0.1,0.9]`.
* `explainable=True`; `_thesis` string returned and attached to bus write.

## 10) Data shapes (canonical, copy-paste)

```yaml
requires:
  trading_signal:
    confidence: float
    action: str
    reason: str
  market_data:
    close: list[number]   # optional; length>=5 enables regime inference
  trades: list[dict]
    - symbol: str
      timestamp: any
      action: str
      pnl: number

provides:
  thesis_analysis:
    current_thesis: str
    thesis_changes: int
    change_frequency: float          # changes/hour
    active_thesis_count: int
    current_confidence: float        # 0..1
    current_performance:
      pnl: float
      trades: int
      confidence: float
    best_thesis: str|null
    worst_thesis: str|null
    total_pnl: float
    total_trades: int
    session_duration_hours: float
    thesis_diversity: float
  thesis_performance:
    <thesis>: { trades: int, pnl: float, confidence: float }
  thesis_alerts:
    - type: str
      severity: str                  # 'medium'|'high'
      message: str
      ...                            # fields like frequency, threshold, thesis, pnl, confidence
```

## 11) Minimal example I/O

```python
inputs = {
  "trading_signal": {"action":"buy","confidence":0.68,"reason":"bullish momentum breakout"},
  "market_data": {"close":[100,101,102,101,103,104]},
  "trades": [
    {"symbol":"XAUUSD","timestamp":1724500100,"action":"buy","pnl":25.5},
    {"symbol":"XAUUSD","timestamp":1724500400,"action":"sell","pnl":-8.0}
  ],
  "timestamp": datetime.datetime.now()
}

result = {
  "thesis_analysis": {...},
  "thesis_performance": {...},
  "thesis_alerts": [...],
  "_thesis": "Thesis evolved/Continuing ... performance +/-$, diversity insight ..."
}
```

## 12) Connections (to fill as we ingest more files)

* **Consumes:** `trading_signal`, `market_data`, `trades` (owners TBD)
* **Provides:** `thesis_analysis`, `thesis_performance`, `thesis_alerts` (consumers TBD)

## 13) Notes / gaps

* `thesis_patterns` declared but not incremented in this file.
* Market regime detection is simplistic (close\[-1] vs close\[-5]); no smoothing or timeframe input.
* Bus write performed only for `thesis_analysis`; other provided keys are returned (publication likely handled by orchestrator).

## 📒 Master ledger (append)

**Module:** MarketDataProvider
**File:** `modules/external/market_data_provider.py`
**Category:** external
**Provides:** `market_data`, `price_data`, `ohlcv_data`, `bid_ask_data`, `prices`, `symbols`, `timestamp`, `technical_indicators`, `indicators`, `volatility_data`, `volatility`, `volatility_index`, `multi_timeframe_data`, `historical_prices`, `market_context`, `trading_session`, `session_type`, `session_canonical`, `data_provider_health`
**Requires:** *(none)*
**Optional Inputs (observed):** *(none; root provider)*
**Side-effects:** reads CSVs from disk; logs to `logs/external/market_data_provider.log`; maintains rolling buffers & indicators; no network I/O
**Voting?** no (`is_voting_member=False`)
**Explainable?** no (`explainable=False`)
**Timeout/Priority:** *(not specified in decorator)*
**Key Methods:** `process`, `_load_data_files`, `_advance_symbol_data`, `_update_technical_indicators`, `_build_snapshot`, `_empty_snapshot`
**State:** `data_files`, `data_iterators`, `current_bars`, `technical_indicators`, `price_buffers`, `current_timestamp`, `trading_session`, `session_type`, `_last_update_ts`, `_update_count`, `_success`, `_fail`, `_proc_times`
**Error paths:** init failure raises; `process()` exception -> returns `_empty_snapshot(error=...)`, increments fail counters
**Consumers (known so far):** AuditingCoordinator (`requires: market_data`), TradeExplanationAuditor (`requires: market_data`), TradeThesisTracker (`requires: market_data`)
**Owners of inputs:** root (no requires)
**Conflicts/Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by **SessionManager** (duplicate providers)

---

# RAW CAPTURE — MarketDataProvider

**Class:** `MarketDataProvider(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name = "MarketDataProvider"`
* `version = "2.1.0"`
* `category = "external"`
* `provides = ["market_data","price_data","ohlcv_data","bid_ask_data","prices","symbols","timestamp","technical_indicators","indicators","volatility_data","volatility","volatility_index","multi_timeframe_data","historical_prices","market_context","trading_session","session_type","session_canonical","data_provider_health"]`
* `requires = []`
* `description = "Offline market data provider that emits only real data from disk. No mock/simulated values."`
* `thesis_required = False`
* `health_monitoring = True`
* `performance_tracking = True`
* `error_handling = True`
* `is_voting_member = False`
* `explainable = False`

## 1) Public API (methods)

* `def __init__(self, config: Optional[Dict[str, Any]] = None)`
* `def _initialize(self) -> None`
* `def _load_data_files(self) -> None`
* `def _find_file_for(self, symbol: str, timeframe: str, data_dir: str) -> Optional[str]`
* `def _initialize_technical_indicators(self) -> None`
* `def _setup_initial_conditions(self) -> None`
* `def _advance_symbol_data(self, symbol: str) -> bool`
* `def _update_technical_indicators(self, symbol: str) -> None`
* `async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`
* `async def process(self, **inputs) -> Dict[str, Any]`
* `def _build_snapshot(self) -> Dict[str, Any]`
* `def _empty_snapshot(self, error: Optional[str] = None) -> Dict[str, Any]`
* `def _bar_with_iso(self, bar: Dict[str, Any]) -> Dict[str, Any]`
* `def _update_session_labels(self) -> None`
* `def _session_canonical(self) -> str`
* `def _is_market_hours(self) -> bool`

## 2) Inputs

* No bus `requires`; reads local CSVs under `cfg.data_directory`.

## 3) Outputs (from `provides` + `_build_snapshot` / `_empty_snapshot`)

```yaml
market_data: { <symbol>: {timestamp, open, high, low, close, volume, bid?, ask?} }
price_data:  { <symbol>: {last, close, open, high, low} }
ohlcv_data:  { <symbol>: {open, high, low, close, volume} }
bid_ask_data:{ <symbol>: {bid, ask, spread} }
prices:      { <symbol>: number }
symbols:     list[str]
timestamp:   ISO datetime string
technical_indicators: { <symbol>: {sma_20, sma_50, rsi, atr, bollinger_upper, bollinger_lower, macd, macd_signal, stochastic} }
indicators:  same as technical_indicators (alias)
volatility_data: { <symbol>: {atr: float, volatility: float} }
volatility:  { <symbol>: float }          # raw ATR value per symbol
volatility_index: float                   # mean normalized volatility across symbols
multi_timeframe_data:
  <symbol>:
    <tf>:
      open|high|low|close|volume: list[20]
      current_bar: {open,high,low,close,volume,bid?,ask?}
      timeframe: str
      bars_available: int
historical_prices: same as multi_timeframe_data (alias)
market_context: { volatility_hint: 'low'|'medium'|'high', market_hours: bool, session_human: str, session_canonical: str }
trading_session: str                      # UI label
session_type: str                         # UI label
session_canonical: 'asian'|'european'|'us'|'closed'
data_provider_health:
  { success_rate: float, avg_processing_time_ms: float, last_update: ISO, update_count: int,
    symbols_loaded: int, timeframes_loaded: int, error?: str }
```

## 4) Side-effects / Logging

* Disk I/O reading CSVs in `_load_data_files`.
* Logs via `RotatingLogger("MarketDataProvider", "logs/external/market_data_provider.log")`.
* Maintains internal performance/health counters (`_success/_fail/_proc_times`).

## 5) Core logic summary

* `_load_data_files`: scans `cfg.data_directory` for patterns like `XAUUSD_H1_features.csv` / `EURUSD_D1.csv` (case-insensitive). Requires `timestamp` or `time` column; coerces to datetime. Accepts close-only datasets: fills O/H/L from `close`, `volume=0`. Cleans NaNs/infs, sorts by timestamp, de-dupes. Builds `data_files[symbol][tf]` and `data_iterators[symbol]`.
* `_advance_symbol_data`: steps the primary TF iterator for a symbol; restarts at beginning on `StopIteration`. Builds `current_bars[symbol]` with OHLCV and optional `bid/ask`; updates buffers/indicators.
* `_update_technical_indicators`: computes SMA(20/50), RSI(14) (simple), ATR(14) from buffers.
* `process`: throttled by `cfg.update_frequency` seconds; advances all supported symbols, updates session labels, returns `_build_snapshot`; on exception returns `_empty_snapshot`.
* `calculate_confidence`: blends availability, freshness (≤60s), and basic quality checks into \[0,1].
* `propose_action`: suggests update & maintenance every 2000 updates; includes `data_quality`.

## 6) Configuration (dataclass `MarketDataConfig`)

* `data_directory: str = "data/processed"`
* `supported_symbols: list[str] = ["XAU/USD","EUR/USD"]`
* `supported_timeframes: list[str] = ["H1","H4","D1"]`
* `primary_timeframe: str = "H4"`
* `update_frequency: float = 1.0`
* `buffer_size: int = 10000`
* `enable_technical_indicators: bool = True`

## 7) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Internal throttling via `update_frequency` and `_last_update_ts`.

## 8) Connections

* **Consumes:** none (root).
* **Provides to (observed so far):** AuditingCoordinator, TradeExplanationAuditor, TradeThesisTracker via `market_data`.
* **Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by `SessionManager`.

## 9) Notes / gaps

* If no valid CSVs found: returns schema-complete but empty snapshot; never fabricates values.
* `volatility_hint` derived from normalized ATR vs price; thresholds fixed in code.
* `historical_prices` is an alias of `multi_timeframe_data` for compatibility.

---

## 📒 Master ledger (append)

**Module:** SessionManager
**File:** `modules/external/session_manager.py`
**Category:** external
**Provides:** `session_metrics`, `session_context`, `system_performance`, `system_health`, `system_alerts`, `trading_session`, `session_type`, `session_canonical`, `time_of_day`
**Requires:** *(none)*
**Optional Inputs (observed):** *(none)*
**Side-effects:** logs to `logs/external/session_manager.log`; maintains counters and alerts list
**Voting?** no (`is_voting_member=False`)
**Explainable?** no (`explainable=False`)
**Timeout/Priority:** *(not specified in decorator)*
**Key Methods:** `process`, `_update_session_labels`, `_session_canonical`, `propose_action`, `calculate_confidence`
**State:** `session_start_ts`, `session_id`, `session_status`, `_success`, `_fail`, `_proc_times`, `system_alerts`, `_last_health_check`, `trading_session`, `session_type`
**Error paths:** `process()` exception → increments fail, appends to `system_alerts`, returns degraded snapshot
**Consumers (TBD):** dashboards/monitors likely; none specified in current files
**Owners of inputs:** root (no requires)
**Conflicts/Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by **MarketDataProvider** (duplicate providers)

---

# RAW CAPTURE — SessionManager

**Class:** `SessionManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name = "SessionManager"`
* `version = "2.0.0"`
* `category = "external"`
* `provides = ["session_metrics","session_context","system_performance","system_health","system_alerts","trading_session","session_type","session_canonical","time_of_day"]`
* `requires = []`
* `description = "Session timing and health context (no fabricated metrics, no duplication with risk/data modules)."`
* `thesis_required = False`
* `health_monitoring = True`
* `performance_tracking = True`
* `error_handling = True`
* `is_voting_member = False`
* `explainable = False`

## 1) Public API (methods)

* `def __init__(self, config: Optional[Dict[str, Any]] = None)`
* `def _initialize(self) -> None`
* `def _update_session_labels(self) -> None`
* `def _session_canonical(self) -> str`
* `async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`
* `async def process(self, **inputs) -> Dict[str, Any]`

## 2) Inputs

* No bus `requires`; uses internal timers/labels only.

## 3) Outputs (from `provides` + `process`)

```yaml
session_metrics:
  session_id: str
  duration: float                # seconds since session start
  status: 'active'|'error'
  start_time: ISO datetime

session_context:
  session_canonical: 'asian'|'european'|'us'|'closed'
  trading_session: 'london'|'new_york'|'sydney'|'tokyo'
  session_type: 'main'|'overlap'|'overnight'

system_performance:
  success_count: int
  failure_count: int
  success_rate: float
  avg_processing_time_ms: float
  last_check: ISO datetime

system_health:
  status: 'healthy'|'degraded'
  alerts: list[dict]             # last 25

system_alerts: list[dict]        # last 25 (same objects as in system_health.alerts)
trading_session: str
session_type: str
session_canonical: 'asian'|'european'|'us'|'closed'
time_of_day: 'HH:MM:SS'
```

## 4) Side-effects / Logging

* Logs via `RotatingLogger("SessionManager", "logs/external/session_manager.log")`.
* Maintains `system_alerts` (append on errors).

## 5) Core logic summary

* `_update_session_labels`: sets human labels (`trading_session`, `session_type`) based on UTC hour.
* `_session_canonical`: maps UTC hour to `'asian'|'european'|'us'|'closed'`.
* `process`: updates labels and timestamps, refreshes `_last_health_check`, builds the snapshot (metrics, context, performance, health, alerts, labels, time\_of\_day), increments success and records proc time; on exception, increments fail, appends an error alert, returns degraded snapshot.

## 6) Configuration (dataclass `SessionConfig`)

* `session_duration: int = 3600`
* `performance_window: int = 500`
* `enable_health_monitoring: bool = True`
* `enable_performance_tracking: bool = True`
* `enable_error_pinpointing: bool = True`

## 7) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* `calculate_confidence`: combines freshness of last health check (≤60s) and counter availability into \[0,1].
* `propose_action`: suggests `reset_session=True` when `duration >= session_duration`.

## 8) Connections

* **Consumes:** none (root).
* **Provides:** session/context/health keys for other modules or UI.
* **Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by `MarketDataProvider`.

## 9) Notes / gaps

* No direct SmartInfoBus `set()` calls in this file; outputs are returned for orchestrator/bus publication.
* Health/performance counters are real-only; no fabricated metrics.

## 📒 Master ledger (append)

**Module:** NewsSentimentModule
**File:** `modules/external/news_sentiment.py`
**Category:** external
**Provides:** `news_sentiment`, `sentiment_confidence`, `news_summary`, `sentiment_trend`
**Requires:** `market_data`, `symbols`, `trading_session`
**Optional Inputs (observed):** `symbol` (from `process(**inputs)`; default chosen from bus)
**Side-effects:** reads/writes SmartInfoBus keys directly; starts optional background monitoring thread; logs to `logs/external/news_sentiment.log`; caches results in-memory; uses env `NEWS_API_KEY`
**Voting?** not specified in decorator (module implements `propose_action`, `calculate_confidence`)
**Explainable?** `thesis_required=True` (returns `_thesis`)
**Timeout/Priority:** *(not specified in decorator)*
**Key Methods:** `process`, `_extract_sentiment_data`, `_process_sentiment_analysis`, `_analyze_sentiment_trend`, `_generate_sentiment_thesis`, `_update_sentiment_smart_bus`, `_fetch_sentiment_from_api`, `_make_api_request`, `_format_declared_outputs`, `get_state/set_state`, `get_health_status`, `propose_action`, `calculate_confidence`
**State:** sentiment + confidence, cache with TTL, API counters, history deque, keyword/category tallies, per-symbol sentiments, performance metrics, circuit breaker, health status, monitoring toggle
**Error paths:** circuit breaker increments on errors; `_handle_sentiment_error` returns safe outputs; health downgraded if API failure rate > 0.5
**Consumers (TBD):** strategy/risk modules may read `news_*` keys (not shown in current files)
**Owners of inputs (known):** `market_data`/`symbols` → MarketDataProvider; `trading_session` → MarketDataProvider **and** SessionManager (duplicate providers)
**Conflicts/Duplicates:** Output **shape mismatch** vs bus writes:

* `news_sentiment`: **process returns float**, but `_initialize` and `_update_sentiment_smart_bus` **set dict** on bus
* `sentiment_confidence`: **process returns float**, bus write is **dict** (`{value, source, min_threshold}`)
* `sentiment_trend`: **process returns string**, bus write is **dict** (`{direction, strength, recent_average}`)

---

# RAW CAPTURE — NewsSentimentModule

**Class:** `NewsSentimentModule(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name = "NewsSentimentModule"`
* `version = "3.0.0"`
* `category = "external"`
* `provides = ["news_sentiment","sentiment_confidence","news_summary","sentiment_trend"]`
* `requires = ["market_data","symbols","trading_session"]`
* `description = "Advanced news sentiment analysis with API integration and SmartInfoBus support"`
* `thesis_required = True`
* `health_monitoring = True`
* `performance_tracking = True`
* `error_handling = True`
* *(no `is_voting_member` field provided)*

## 1) Public API (methods)

* `__init__(config: Optional[SentimentConfig] = None, genome: Optional[Dict[str, Any]] = None, **kwargs)`
* `_initialize_advanced_systems()` — bus/logger/error/perf/circuit breaker/health setup
* `_initialize_genome_parameters(genome: Optional[Dict[str, Any]])`
* `_initialize_sentiment_state()` — caches, counters, history
* `_start_monitoring()` — spawns background thread (30s loop) to update health & expire cache
* `_initialize()` — initial bus seed for `news_sentiment`
* `async process(**inputs) -> Dict[str, Any]` — main pipeline
* `async _extract_sentiment_data(**inputs) -> Optional[Dict[str, Any]]`
* `async _process_sentiment_analysis(sentiment_data: Dict[str, Any]) -> Dict[str, Any]`
* `_get_cached_sentiment(symbol: str) -> Optional[Dict[str, Any]]`
* `_cache_sentiment(symbol: str, sentiment: float, confidence: float) -> None`
* `async _fetch_sentiment_from_api(symbol: str) -> Dict[str, Any]`
* `_build_query_for_symbol(symbol: str) -> str`
* `async _make_api_request(query: str) -> Dict[str, Any]` — simulated HTTP
* `async _analyze_sentiment_trend() -> Dict[str, Any]`
* `async _generate_sentiment_thesis(sentiment_data: Dict[str, Any], sentiment_result: Dict[str, Any]) -> str`
* `async _update_sentiment_smart_bus(sentiment_result: Dict[str, Any], thesis: str) -> None`
* `_cleanup_expired_cache() -> None`
* `_update_sentiment_health() -> None`
* `_format_declared_outputs(sentiment_result: Dict[str, Any], thesis: str) -> Dict[str, Any]`
* `async _handle_no_data_fallback() -> Dict[str, Any]`
* `async _handle_sentiment_error(error: Exception, start_time: float) -> Dict[str, Any]`
* `_record_success(processing_time_ms: float) -> None`
* `_record_failure(error: Exception) -> None`
* `get_state() -> Dict[str, Any]`
* `set_state(state: Dict[str, Any]) -> None`
* `get_health_status() -> Dict[str, Any]`
* `stop_monitoring() -> None`
* `set_sentiment(value: float) -> None` — manual override; **starts monitoring thread**
* `async propose_action(**inputs) -> Dict[str, Any]`
* `async calculate_confidence(action: Dict[str, Any], **inputs) -> float`
* `confidence(obs: Any = None, **kwargs) -> float` (legacy)

## 2) Configuration (dataclass `SentimentConfig`)

* `enabled: bool = False`
* `default_sentiment: float = 0.0`
* `cache_ttl: int = 60` (seconds)
* `max_retries: int = 2`
* `timeout: float = 10.0`
* Performance thresholds: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_confidence=0.3`
  **Env:** `NEWS_API_KEY` (empty string → default/fallback path)

## 3) Inputs (from bus + optional `symbol`)

* **Bus reads** (inside `_extract_sentiment_data`):

  * `market_data` (dict)
  * `symbols` (list or scalar)
  * `trading_session` (str/dict; used only for context)
* `symbol` (optional in `process(**inputs)`): if absent, picks first from `symbols`; else from `market_data` fallback `'EURUSD'`.

## 4) Outputs (declared via return of `process`)

`_format_declared_outputs` returns strictly:

```yaml
news_sentiment: float                # sentiment value in [-1, 1]
sentiment_confidence: float          # 0..1
news_summary:
  total_requests: int
  cache_hits: int
  api_successes: int
  api_failures: int
  enabled: bool
  api_configured: bool
sentiment_trend: str                 # 'improving'|'declining'|'stable' or 'insufficient_data'
_thesis: str
```

**Note:** The module also performs **direct bus writes** with **different shapes**:

* `news_sentiment` → **dict** `{sentiment, confidence, enabled, api_configured, last_updated}`
* `sentiment_confidence` → **dict** `{value, source, min_threshold}`
* `news_summary` → **dict** `{total_requests, cache_hits, api_successes, api_failures}`
* `sentiment_trend` → **dict** `{direction, strength, recent_average}`

## 5) Core pipeline (inside `process`)

1. `_extract_sentiment_data()` from bus (+ choose `symbol`)
2. If no data → `_handle_no_data_fallback()` (returns neutral/default)
3. `_process_sentiment_analysis()`:

   * If `enabled=False` → use `default_sentiment`, `confidence=0.0`, `source='default'`
   * Else check cache; if fresh → use cached
   * Else if `api_key` present → `_fetch_sentiment_from_api()` with retries; cache result
     If retries exhausted or error → fall back to default neutral (`source='fallback'|'error'`)
   * Updates `latest_sentiment`, `sentiment_confidence`, and performance counters
4. `_analyze_sentiment_trend()`:

   * If history length < 5 → `'insufficient_data'`
   * Else linear fit slope over last 5; classify `'improving'|'declining'|'stable'`
5. `_generate_sentiment_thesis()` — human-readable explanation string
6. `_update_sentiment_smart_bus()` — writes *dict-shaped* keys to bus
7. `_record_success()` and return formatted declared outputs

**Error path:** any exception → `_handle_sentiment_error()`:

* increments circuit breaker, logs structured error, records failure, returns safe defaults via `_format_declared_outputs`.

## 6) Scoring & models

* **Sentiment value**: float in \[-1, 1]; random synth in `_make_api_request` (simulated API)
* **Confidence**: `0.5 + |sentiment| * 0.5` (simulated); default `0.0` when disabled/no API
* **Trend**: sign and magnitude of slope over last-5 sentiments (`np.polyfit`)

## 7) Monitoring, health & circuit breaker

* `_start_monitoring()` thread (daemon):

  * Every 30s: `_update_sentiment_health()` and `_cleanup_expired_cache()`
  * **Started only by `set_sentiment()` (manual override)** in current file
* Health:

  * `_health_status` `'healthy'` unless API failure rate > 0.5 → `'warning'`
  * `get_health_status()` returns: status, last\_check, circuit breaker state, api\_configured, enabled, cache\_size
* Circuit breaker:

  * `failures`, `last_failure`, `state ('CLOSED'|'OPEN')`, `threshold` (from config)
  * **Opened** when consecutive failures ≥ threshold; **reset** to CLOSED on `_record_success()`
  * Not used to gate further calls (no short-circuiting in code)

## 8) Caching

* `_cache: {symbol: {sentiment, confidence, timestamp}}`
* TTL: `genome["cache_ttl"]` seconds
* Eviction: `_cleanup_expired_cache()` (monitor thread) or on stale check inside `_get_cached_sentiment`

## 9) Logging & performance

* `RotatingLogger(..., operator_mode=True, plain_english=True)`
* Operator messages via `format_operator_message`
* Performance via `PerformanceTracker.record_metric('NewsSentimentModule', 'sentiment_analysis', ...)`

## 10) State (persistence)

`get_state()` / `set_state()` cover:

* `latest_sentiment`, `sentiment_confidence`, `_cache`, `genome`, `_api_call_count`, `_api_failures`,
* `_sentiment_performance`, `circuit_breaker`, `_health_status`

## 11) Data shapes (canonical, copy-paste)

```yaml
requires (bus reads):
  market_data: dict
  symbols: list[str] | str
  trading_session: any

provides (returned from process):
  news_sentiment: float            # [-1, 1]
  sentiment_confidence: float      # [0, 1]
  news_summary:
    total_requests: int
    cache_hits: int
    api_successes: int
    api_failures: int
    enabled: bool
    api_configured: bool
  sentiment_trend: str             # 'improving'|'declining'|'stable'|'insufficient_data'
  _thesis: str
```

## 12) Connections

* **Consumes:** `market_data`, `symbols` → **MarketDataProvider**; `trading_session` → **MarketDataProvider**/**SessionManager** (duplicate providers)
* **Provides:** sentiment keys for downstream strategy/risk/monitoring modules (not shown in current files)

## 13) Notes / gaps

* **Shape divergence** between returned outputs and direct bus writes for `news_sentiment`, `sentiment_confidence`, `sentiment_trend` (see conflicts above).
* `_sentiment_history` is referenced for trend but **never appended** in this file; trend may stay `'insufficient_data'` unless history is updated elsewhere.
* Monitoring thread (`_start_monitoring`) **not started in init/process**; only invoked in `set_sentiment()` (manual path).


## 📒 Master ledger (append)

**Module:** AdvancedFeatureEngine
**File:** `modules/features/advanced_feature_engine.py`
**Category:** features
**Provides:** `advanced_features`, `features` (alias), `feature_analysis`, `feature_thesis` (+ returns hidden `_thesis`)
**Requires:** `price_data` (hard requirement), with fallbacks via InfoBus: `historical_prices` → `ohlcv_data` → `market_data`
**Optional direct inputs supported:** `prices`, `price`, `close`, `price_series`
**Side-effects:** Writes declared keys to SmartInfoBus; background health/perf monitors (async tasks)
**Voting?** `is_voting_member=False`
**Explainable?** `thesis_required=True`
**Timeout/Priority:** `timeout_ms=120`, hot reload enabled

---

# RAW CAPTURE — AdvancedFeatureEngine

**Config (`FeatureEngineConfig`)**

* `window_sizes: List[int]` (default `[7, 14, 28, 56]`)
* `max_buffer_size: int` (default `1000`)
* `enable_neural_processing: bool` (placeholder, default `False`)
* `enable_health_monitoring: bool` (`True`)
* `enable_performance_tracking: bool` (`True`)
* `enable_error_pinpointing: bool` (`True`)
* `enable_english_explanations: bool` (`True`)
* `circuit_breaker_threshold: int` (`5`)

**Initialization / State**

* `smart_bus = InfoBusManager.get_instance()`
* `window_sizes` sorted; `out_dim = len(window_sizes)*6 + 6`
* Buffers: `price_buffer(maxlen=max_buffer_size)`, `feature_buffer(maxlen=1000)`
* Last outputs: `last_features: np.ndarray(out_dim)`, `feature_quality_score: float`
* Stats: `feature_stats = {total_extractions, successful_extractions, failed_extractions, avg_extraction_time_ms, avg_feature_quality, price_points_processed}`
* Health: `health_metrics = {last_health_check, health_score, issues_detected[], performance_trend}`
* Circuit breaker: `{failures, last_failure, state('CLOSED'|'HALF_OPEN'|'OPEN'), threshold}`

**Main pipeline (`process`)**

1. **Breaker gate:** `_check_circuit_breaker()`; if open and cooldown not passed → returns formatted fallback.
2. **Input extraction:** `_extract_market_data(**inputs)`

   * Prefers `price_data` from inputs: expects `dict[symbol] -> {'close': float}`
   * Fallbacks (from InfoBus, in order): `historical_prices` (uses `close` arrays or `current_bar.close`) → `ohlcv_data` → `market_data` (keys containing “price” list or nested `{close}`).
   * Also accepts loose inputs: `prices | price | close | price_series`.
   * Prices validated via `_validate_prices`: finite, positive; optional 3σ trim for outliers (if len>10).
   * Raises `ValueError` if no usable prices.
3. **Feature extraction:** `_process_features_with_monitoring(market_data)`

   * Extends `price_buffer`; computes `feats = _extract_comprehensive_features(prices)`.
   * Quality score via `_calculate_feature_quality(feats)`; caches `last_features`; append to `feature_buffer`.
   * Stats update and English explanation string.
   * Returns payload: `{raw_features: np.ndarray, quality_score: float, extraction_time_ms, buffer_size, feature_count, explanation}`
4. **Thesis:** `_generate_feature_thesis(features_payload, market_data)` (plain-English report with current price, % change, counts, and recommendations).
5. **Bus writes:** `_update_bus(features_payload, thesis)` publishes only declared keys.
6. **Success bookkeeping:** `_record_success()`; optional `PerformanceTracker`.
7. **Return (contract):** `_format_declared_outputs(...)` → declared keys + `_thesis` and optional extras.

**Feature definitions**

* Per window (6 stats/window): `[mean, std, return, range, up_ratio, n]`

  * `return = (last - first)/first`
  * `up_ratio = mean(diffs>0)`
* Global (6 stats): `[last_price, global_mean, global_std, global_range, pos_step_ratio, n]`
* Total dim: `len(window_sizes)*6 + 6`

**Quality scoring (`_calculate_feature_quality`)**

* Guards: non-array/empty → `0`; any non-finite → `0`
* Detects flat data: near-zero per-window std/ret/range & global std/range near 0 & pos\_step \~0 or \~1 → returns `30.0`
* Zero variance of feature vector → `25.0`
* Very small span → `35.0`
* Baseline `85.0`, penalize large magnitudes (`>1e6` −15), modest std (0.05–250) `+5`
* Clamped to `[0,100]`

**Monitoring / background**

* `_start_monitoring()` schedules (if event loop present):

  * `_health_monitoring_loop()` every **30s**: updates `health_score`, issues list (breaker open, buffer near full, avg time high, avg quality low)
  * `_performance_monitoring_loop()` every **60s**: `PerformanceTracker.record_metric(...)`

**Errors & breaker**

* `_handle_processing_error` logs, updates breaker and health; returns formatted fallback (zeros vector, quality 0, `_thesis` with error).
* Breaker opens when `failures >= threshold`; half-open after 60s; closes on success.

**Bus writes (single-writer, declared keys only)**

* `advanced_features` → `{raw_features: List[float], quality_score: float, extraction_time_ms: float, timestamp: float}`
* `features` (alias) → `{raw_features: List[float], quality_score: float}`
* `feature_analysis` → `{explanation: str|None, buffer_status{current_size,max_size,utilization}, statistics: feature_stats}`
* `feature_thesis` → `str`
* Hidden `_thesis` returned in the dict (orchestrator requirement)

**Returned shapes (contract)**

```yaml
advanced_features:
  raw_features: list[float]           # length == out_dim
  quality_score: float                # 0..100
  extraction_time_ms: float
  buffer_size: int
  feature_count: int
features:
  raw_features: list[float]
  quality_score: float
feature_analysis:
  explanation: str | null
  statistics: dict
  buffer_status:
    current_size: int
    max_size: int
    utilization: float
feature_thesis: str
_thesis: str
# Extras that may also be present:
success: bool
processing_time_ms: float
```

**Connections**

* **Consumes:** `price_data` (MarketDataProvider). Fallbacks use `historical_prices/ohlcv_data/market_data` (MarketDataProvider).
* **Provides to:** MultiScaleFeatureEngine (reads `advanced_features`); any strategy/risk modules needing vectorized features.

**Notes / gaps / pitfalls**

* Requires a running event loop to start monitors; otherwise defers (safe).
* Circuit breaker only gates processing; does not publish errors to bus (returns fallback).
* English explainer/system utilities used for human reports; disabling removes narratives but not core features.

---

**Module:** MultiScaleFeatureEngine
**File:** `modules/features/multiscale_feature_engine.py`
**Category:** features
**Provides:** `multiscale_features`, `neural_embeddings`, `attention_weights`, `feature_fusion` (+ returns hidden `_thesis`)
**Requires:** `advanced_features` (prefers InfoBus; can use injected AFE instance)
**Optional upstream keys consumed (if present):** `advanced_features_{TF}` for TF in cfg.timeframes (e.g., `advanced_features_H1`)
**Side-effects:** Writes declared keys to SmartInfoBus; GPU usage optional; background neural/gpu monitors
**Voting?** `is_voting_member=False`
**Explainable?** `thesis_required=True`
**Timeout/Priority:** `timeout_ms=180`, hot reload enabled

---

# RAW CAPTURE — MultiScaleFeatureEngine

**Config (`MultiScaleConfig`)**

* `embed_dim: int = 64` (output embedding size)
* `num_attention_heads: int = 4`
* `dropout_rate: float = 0.10`
* `enable_gpu: bool = True`
* `feature_fusion_method: str = "attention" | "concat" | "weighted"` (used as label; current path uses attention+MLP)
* `timeframes: List[str]` (default `["H1","H4","D1"]`)
* `assumed_input_dim: int = 256` (used if upstream dim undiscoverable)

**Initialization / State**

* `smart_bus = InfoBusManager.get_instance()`
* `device = "cuda"` if available & enabled else `"cpu"`
* **Input dim discovery (`_discover_input_dim`):**

  1. If injected `afe` has `out_dim` → use it
  2. Else read `advanced_features.raw_features` length from InfoBus
  3. Else fallback `assumed_input_dim`
* Networks (`_build_networks(input_dim)`):

  * Per-TF processor: `Linear(input_dim→E)→ReLU→LayerNorm→Dropout`
  * Fusion MLP over concatenated TF embeddings: `Linear(E*T→2E)→ReLU→LayerNorm→Dropout→Linear(2E→E)→ReLU→Linear(E→E)`
  * Attention fusion (`AttentionFeatureFusion`): projects to `E`, self-attention (`nn.MultiheadAttention`), residual+LayerNorm+MLP
* State:

  * `last_embedding: np.ndarray(E)`
  * `attention_weights_history: deque(maxlen=100)`
  * `embedding_history: deque(maxlen=500)`
  * Stats `neural_stats = {total_forward_passes, successful_passes, failed_passes, avg_forward_time_ms, avg_attention_entropy, gpu_memory_usage_mb}`
  * Health `neural_health = {model_health_score, gradient_health, attention_quality, embedding_quality, last_neural_check}`
  * Breaker `neural_circuit_breaker = {failures, last_failure, state('CLOSED'|'HALF_OPEN'|'OPEN'), threshold=3}`
* Background:

  * `_neural_health_monitoring_loop()` every **60s**
  * `_gpu_monitoring_loop()` every **30s** if CUDA present

**Main pipeline (`process`)**

1. **Breaker gate:** `_check_neural_circuit_breaker()`; if open and cooldown not passed → returns neural fallback (last/zeros).
2. **Upstream features:** `_get_advanced_features(**inputs)`

   * If injected AFE instance present: `await afe.process(**inputs)` → extract `advanced_features.raw_features`
   * Else from InfoBus: `advanced_features.raw_features`
   * If none, generate deterministic synthetic vector (`np.random.normal(...)`) of `input_dim`
   * **Hot-swap**: `_maybe_rebuild_for(new_dim)` rebuilds all networks if feature length changes at runtime.
3. **Timeframe shaping:** `_process_multiscale_features(afe_payload)`

   * Reads ONLY real per-TF vectors from InfoBus keys `advanced_features_{TF}` (e.g., `advanced_features_H1`) and validates shape equals base vector length.
   * Computes safe pairwise correlations between TF vectors (`_safe_corr` guards mean/std and size).
   * Returns `{timeframe_features: Dict[tf→np.ndarray], correlations: Dict["A_B"→float], base_features: np.ndarray, processing_time_ms}`
   * If no TF data found → returns empty `timeframe_features`/`correlations` (no fabricated TF vectors).
4. **Neural forward:** `_neural_forward(ms_result)`

   * Per-TF processing on device; concatenate embeddings; fusion MLP; self-attention across TFs; average fused+attended → `embedding[E]`
   * Returns `{embeddings: np.ndarray[B,E], attention_weights: np.ndarray[B,H,T,T], processed_features: Dict[tf→np.ndarray], forward_time_ms, attention_entropy}`
   * Updates `last_embedding`, histories, and stats
5. **Thesis:** `_generate_neural_thesis(...)` (human-readable: TF count, dim, attention entropy, speed, quality score)
6. **Bus writes:** `_update_bus(ms_result, nn_result, thesis)` (declared keys only)
7. **Success bookkeeping:** `_record_neural_success()`
8. **Return (contract):** `_format_declared_outputs(...)` → declared keys + `_thesis` and timing extras

**Neural quality / metrics**

* Attention entropy computed over weights (normalized).
* Embedding quality (`_assess_embedding_quality`): checks finite/non-flat; baseline 90 with penalties if |val|>10; bonus for std in \[0.1,5.0]; 0..100.
* Stats update keeps rolling averages; GPU memory sampled if CUDA.

**Errors & breaker**

* `_handle_neural_error` logs, updates breaker and health; returns fallback response (`last_embedding` or zeros; zeroed attention).
* Breaker opens when failures ≥ 3; half-open after 120s; closes on success.

**Bus writes (declared keys only)**

* `multiscale_features` → `{correlations: Dict[str,float], timeframe_features: Dict[tf→list[float]], processing_time_ms: float}`
* `neural_embeddings` → `{embeddings: list[float] (or nested if B>1), dimensions: list[int], device: str, timestamp: float}`
* `attention_weights` → `{weights: nested list, entropy: float, num_heads: int, timeframes: list[str]}`
* `feature_fusion` → `{processed_features: Dict[tf→list[float]], fusion_method: str, processing_time_ms: float}`

**Returned shapes (contract)**

```yaml
multiscale_features:
  correlations: dict[str, float]          # e.g., "H1_H4": 0.12
  timeframe_features: dict[str, list[float]]
  processing_time_ms: float
neural_embeddings: list[float]            # flattened embedding (E) if B==1
attention_weights: list                   # nested list [H, T, T] or []
feature_fusion:
  processed_features: dict[str, list[float]]
  fusion_method: str
  processing_time_ms: float
_thesis: str
# Extras that may also be present:
processing_time_ms: float
device_used: str
success: bool
```

**Shape caveat (bus vs return)**

* **Return**: `neural_embeddings` is a **list\[float]** and `attention_weights` is a **nested list**.
* **Bus write**: `neural_embeddings` and `attention_weights` are **dicts** with metadata (`dimensions`, `device`, etc.).
* If a downstream module reads directly from the bus, it must expect the **dict** shapes; if it chains off the **returned** dict, it will get the simpler list shapes.

**Connections**

* **Consumes:** `advanced_features` (AdvancedFeatureEngine). Optionally `advanced_features_{TF}` (whoever publishes per-TF vectors; could be multiple AFE instances keyed by TF).
* **Provides to:** downstream neural/decision modules; can be used by strategies expecting embeddings/attention diagnostics.

**Notes / gaps / pitfalls**

* If no per-TF vectors are published (no `advanced_features_H1` etc.), `timeframe_features` stays empty; neural forward path expects TFs present—current code populates `processed` only for configured TFs, so ensure those TF keys exist **or** guard before `_neural_forward` (present code assumes TF keys exist when building `processed`; if none, earlier `_process_multiscale_features` returns empty, but `_neural_forward` indexes `ms_result["timeframe_features"][tf]` → will error if missing).
* Dynamic input-dim hot-swap rebuilds networks live; safe but may momentarily change embedding distribution.
* GPU monitor only runs if CUDA available; attention entropy stored but not aggregated into `avg_attention_entropy` (present but not updated elsewhere).

---

## Data-flow links (between the two)

* **MarketDataProvider → AdvancedFeatureEngine**: `price_data` / `historical_prices` / `ohlcv_data` / `market_data`
* **AdvancedFeatureEngine → MultiScaleFeatureEngine**: `advanced_features.raw_features` length defines `input_dim`; optional per-TF keys `advanced_features_{TF}` enable correlations & attention over TFs.
* **Both publish to InfoBus** with **single-writer discipline** over their own keys.


awesome — got both. here’s the clean capture + ledger entries in the same format as before.

---

## 📒 Master ledger (append)

**Module:** FractalRegimeConfirmation
**File:** `modules/market/fractal_regime_confirmation.py`
**Category:** market
**Provides:** `market_regime`, `regime_strength`, `trend_direction`, `fractal_metrics`, `regime_data`, `symbols`, `timestamps`
**Requires:** `prices`, `step_idx`, `volatility_level` *(best-effort; also supports a structured `data_dict/current_step` path)*
**Optional Inputs (observed):** `data_dict`, `current_step`, `theme_detector` (legacy path)
**Side-effects:** writes declared keys to SmartInfoBus; logs operator messages; keeps internal histories; background health task (if loop)
**Voting?** no (`is_voting_member=False`; mixin includes voting helpers but not a voter)
**Explainable?** yes (`thesis_required=True`, returns `_thesis` and mirrors it as `thesis`)
**Timeout/Priority:** `timeout_ms=180`, hot reload enabled
**Key Methods:** `process`, `_extract_market_data_comprehensive`, `_process_regime_detection` (+ format variants), `_compute_fractal_metrics_robust` (H, VR, WE), `_process_regime_signals` (hysteresis), `_publish_bus`, `propose_action`, `calculate_confidence`, `get_regime_analysis_report`
**State:** regime label/strength, trend, stability score, theme score, deque buffers for scores/regime history/metrics; circuit breaker; metrics aggregates; last symbols/timestamps; stale prices cache
**Error paths:** circuit breaker (OPEN after 3 fails, HALF\_OPEN after 120s), safe fallbacks with cached state, detailed logging via `ErrorPinpointer`
**Consumers (TBD):** downstream strategy/risk selection, dashboards
**Owners of inputs (known):** `prices`/`step_idx` likely from **MarketDataProvider** / orchestrator
**Conflicts/Duplicates:** none on keys; overlaps conceptually with other “regime/session” providers but uses distinct keys

---

**Module:** LiquidityHeatmapLayer
**File:** `modules/market/liquidity_heatmap_layer.py`
**Category:** market
**Provides:** `liquidity_score`, `market_depth`, `spread_analysis`, `liquidity_prediction`, `trading_sessions`, `session_data`, `liquidity_thesis`, `liquidity_capabilities`
**Requires:** `bid_ask_data`, `price_data`, `prices`
**Optional Inputs (observed):** `market_data` (loose dict with arrays for prices/volumes/spreads/depth)
**Side-effects:** writes declared keys to SmartInfoBus; initializes and runs PyTorch LSTM+Attention; background monitoring; logs operator messages; performance metrics; circuit breaker
**Voting?** no (`is_voting_member=False`)
**Explainable?** yes (`thesis_required=True`, returns `_thesis` exposed as `liquidity_thesis`)
**Timeout/Priority:** *(not stated at decorator level)*; health/perf tracking enabled
**Key Methods:** `process`, `_extract_market_data`, `_analyze_liquidity`, `_neural_liquidity_prediction`, `_generate_liquidity_thesis`, `_update_liquidity_smart_bus`, `propose_action`, `calculate_confidence`, `get_health_status`
**State:** price/spread/depth/volume deques; sequence buffer; training buffer (unused for learning); current liquidity/spread/depth; health & stats; neural circuit breaker
**Error paths:** circuit breaker (OPEN ≥3 fails, HALF\_OPEN after 120s); structured fallback payload; anomaly warnings in monitor
**Consumers (TBD):** execution sizing, slippage/impact models, dashboards
**Owners of inputs (known):** **MarketDataProvider** (for `prices`/`price_data`/`bid_ask_data`)
**Conflicts/Duplicates:** **Potential symbol canonicalization mismatch** — expects `"EURUSD"/"XAUUSD"` codes when reading `prices/price_data`; ensure upstream uses same keys or adapt

---

## RAW CAPTURE — FractalRegimeConfirmation

**Class:** `FractalRegimeConfirmation(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin)`
**Decorator metadata (@module):**

* `name="FractalRegimeConfirmation"`, `version="3.1.0"`, `category="market"`
* `provides=["market_regime","regime_strength","trend_direction","fractal_metrics","regime_data","symbols","timestamps"]`
* `requires=["prices","step_idx","volatility_level"]`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`
* `is_voting_member=False`, `hot_reload=True`, `timeout_ms=180`

### 1) Public API

* `async process(**inputs) -> dict` — main loop (cb gate → data extract → metrics → hysteresis → bus → return)
* `step(...) -> (label, strength)` — legacy sync step (no bus writes)
* `propose_action(...)`, `calculate_confidence(...)`, `confidence(...)`
* Reporting/obs: `get_observation_components()`, `get_regime_analysis_report()`

### 2) Inputs the code accepts

* **Preferred structured (legacy):** `data_dict` (map `{symbol: {TF: DataFrame}}`) + `current_step` (+ optional `theme_detector`)
* **Current InfoBus path:** `prices` (dict of last prices), `step_idx` (int), `volatility_level` (str)
* Fallback: last known `prices` cache or synthetic generator (if nothing available)

### 3) Outputs (contract)

```yaml
market_regime: "noise" | "volatile" | "trending"
regime_strength: float  # [0..1] (internally can exceed 1 before clamp)
trend_direction: float  # [-1..1]
fractal_metrics: { H: float, VR: float, WE: float }
regime_data:
  id: 0|1|2|3        # {noise:0, range:1, trend/trending:2, volatile:3}
  market_regime: str
  regime_strength: float
  trend_direction: float
symbols: string[]     # last used instruments
timestamps: string[]  # window slice if DF path used
thesis: string
_thesis: string       # same as thesis
# extras may include: success, processing_time_ms
```

### 4) Side-effects / Bus writes

* Publishes **declared keys only**: `market_regime`, `regime_strength`, `trend_direction`, `regime_data`, `symbols`, `timestamps` (with operator-style `thesis` strings).

### 5) Core logic

* **Metrics** on a price series (`ts`):

  * Hurst (`_hurst_enhanced`): log–log std of differences over multi-lag; robust clamps; returns \~\[0,1].
  * Variance Ratio (`_var_ratio_enhanced`): `var(k-step)/k` vs `var(1-step)` \~ \[0.1,10].
  * Wavelet Energy (`_wavelet_energy_enhanced`): detail energy % via `pywt` db4 up to level 2.
* **Score → strength**: `score = 0.40*H + 0.30*VR + 0.30*WE`, smoothed by 3-point median of recent scores. Multiplied by `theme_conf` (0.5..1.0). Output `regime_strength` is **clamped** \[0,1].
* **Hysteresis** (to avoid flapping):

  * thresholds: noise→volatile `0.30`; volatile→trending `0.60`; back edges `0.20` / `0.50`
  * extra stability guard: if last 5 regimes include >2 unique and strength near 0.5, keep old label.
* **Trend direction**: mean(last 5) vs mean(first 5) or (last-first)/first → mapped to \[-1,1].
* **Stability score**: from regime diversity over last 10 samples → 0..100.

### 6) Health / circuit breaker

* CB states: `CLOSED` → `OPEN` at ≥3 consecutive failures → `HALF_OPEN` after 120s; on success resets.
* On CB-open or data-missing: returns cached state + minimal metrics; logs via `ErrorPinpointer` and `EnglishExplainer`.

### 7) Notable details / gaps

* `requires` lists `step_idx`/`volatility_level`, but extraction mostly relies on InfoBus reads; direct `inputs` path is the legacy `data_dict/current_step`.
* `fractal_metrics` filter in `_format_declared_outputs` keeps only finite numeric values.
* Theme integration hook (`theme_detector`) adjusts strength (0.5–1.0 multiplier) but defaults to `1.0`.

---

## RAW CAPTURE — LiquidityHeatmapLayer

**Class:** `LiquidityHeatmapLayer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name="LiquidityHeatmapLayer"`, `version="3.0.2"`, `category="market"`
* `provides=["liquidity_score","market_depth","spread_analysis","liquidity_prediction","trading_sessions","session_data","liquidity_thesis","liquidity_capabilities"]`
* `requires=["bid_ask_data","price_data","prices"]`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Public API

* `async process(**inputs) -> dict` (cb gate → data extract → analytics → NN prediction → sessions → bus → return)
* `propose_action(...)` (recommends trade posture & size), `calculate_confidence(...)`
* Health/Reports: `get_health_status()`, `get_liquidity_performance_report()`
* State I/O: `get_state()/set_state()`

### 2) Inputs the code accepts

* **InfoBus:** `prices`, `price_data`, `bid_ask_data`

  * ⚠️ Canonicalization: it converts `"EUR/USD"` → `"EURUSD"` when querying dictionaries. Upstream should match this for hits.
* **Optional direct:** `market_data` dict containing arrays for `prices`, `volumes`, `bid_ask_spreads`, and `market_depth {bids,asks}`.
* **Synthetic fallback** if nothing present.

### 3) Outputs (contract)

```yaml
liquidity_score: float            # [0..1]
market_depth:
  current_depth: float
  analysis: dict                 # {average_depth, stability_score, condition, trend, status}
  condition: "deep"|"moderate"|"shallow"|...
spread_analysis:
  current_spread: float
  analysis: dict                 # {average_spread, spread_volatility, condition, trend, status}
  condition: "tight"|"normal"|"wide"|...
liquidity_prediction:
  predictions: {liquidity_score?: float, depth?: float, spread?: float}
  confidence: float              # [0..1]
  horizon_steps: int
  status: "success"|"insufficient_sequence_data"|"error"|"unavailable"
trading_sessions:                 # computed from UTC hour + weekend/rollover rules
  {asian: bool, european: bool, american: bool, rollover: bool, weekend: bool, active: str}
session_data:
  {active_session: str, utc_time: ISO, windows_utc: {...}, liquidity_bias: float}
liquidity_thesis: string
liquidity_capabilities:
  {prediction_horizon, sequence_length, device, depth_levels, neural_model}
_thesis: string
# extras may include: success, processing_time_ms
```

### 4) Side-effects / Bus writes

* Writes **all provided keys** (score, depth, spread, prediction, sessions, session\_data, thesis, capabilities) with human theses.

### 5) Core logic

* **Liquidity score** (0..1) combines:

  * **Spread** (lower→better; normalized vs 0.001) — weight 0.4
  * **Depth** (higher→better; normalized by 10k) — weight 0.4
  * **Short-term volatility** of price (lower→better) — weight 0.2
* **Spread analysis:** avg/std over recent spreads; condition from thresholds:

  * tight if `avg < low_liquidity_threshold * 0.001` (default low=0.3)
  * wide if `avg > high_liquidity_threshold * 0.001` (default high=0.8)
* **Depth analysis:** avg/std; stability score = `1 - (std/avg)`; condition via absolute depth cutoffs (`>50k deep`, `<10k shallow`).
* **Volumes:** avg/trend/volatility if provided.
* **Neural prediction:** LSTM(+MultiheadAttention) over a sequence of 4 features

  * features per step: `[current_spread, depth/10000, liquidity_score, volume_volatility]`
  * needs `sequence_length` samples (default 20). If fewer → `status="insufficient_sequence_data"`.
  * returns predicted `liquidity_score/depth/spread` + confidence blended from recent performance & health (0..1).
  * Model is **inference-only** here (no live training loop).

### 6) Sessions snapshot

* Derives active session from UTC (asian/european/american/rollover/weekend) and provides a `liquidity_bias` hint.

### 7) Health / circuit breaker

* Neural CB: `OPEN` after ≥3 failures; `HALF_OPEN` after 120s; success closes CB.
* Health monitor adjusts `data_quality_score` and `model_health_score`; anomaly checker warns on extreme spreads, low depth, CB open.

### 8) Notable details / gaps

* **Symbol keys:** Expects `"EURUSD"/"XAUUSD"` codes when querying InfoBus dicts; make sure upstream uses those (or extend matcher).
* Predictions depend on **untrained** network weights unless you wire in a training loop — treat outputs as diagnostic unless trained.
* `_format_declared_outputs` always returns `liquidity_prediction` as a dict; safe even on errors.

---

## Data shapes (copy-paste)

### FractalRegimeConfirmation

```yaml
requires:
  prices: dict[str, float]          # e.g., {"EURUSD": 1.1012, "XAUUSD": 2350.1}
  step_idx: int                     # optional via InfoBus
  volatility_level: str             # optional via InfoBus
# OR legacy:
  data_dict: dict[str, dict[str, DataFrame]]
  current_step: int

provides:
  market_regime: str                # noise|volatile|trending
  regime_strength: float            # [0..1]
  trend_direction: float            # [-1..1]
  fractal_metrics: {H: float, VR: float, WE: float}
  regime_data: {id: int, market_regime: str, regime_strength: float, trend_direction: float}
  symbols: list[str]
  timestamps: list[str]
  _thesis: str
```

### LiquidityHeatmapLayer

```yaml
requires:
  prices: dict[str, float]              # expects keys like "EURUSD"/"XAUUSD"
  price_data: dict[str, {close: float}]
  bid_ask_data: dict[str, {spread: float}]
# optional direct:
  market_data:
    prices: list[float]
    volumes: list[float]
    bid_ask_spreads: list[float]
    market_depth: {bids: list[[price, size]], asks: list[[price, size]]}

provides:
  liquidity_score: float                # [0..1]
  market_depth: {current_depth: float, analysis: dict, condition: str}
  spread_analysis: {current_spread: float, analysis: dict, condition: str}
  liquidity_prediction: {predictions: dict, confidence: float, horizon_steps: int, status: str}
  trading_sessions: dict
  session_data: dict
  liquidity_thesis: str
  liquidity_capabilities: dict
  _thesis: str
```



nice — got it. here’s the clean capture + ledger entry for **MarketThemeDetector** in the same style.

---

## 📒 Master ledger (append)

**Module:** MarketThemeDetector
**File:** `modules/market/market_theme_detector.py`
**Category:** market
**Provides:** `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`, `theme_analysis`, `theme_detection`, `theme_detector_status`, `theme_detector_health`, `theme_model_quality`
**Requires:** `market_data`, `price_data`, `technical_indicators`, `historical_prices`, `multi_timeframe_data`, `macro_data` *(all best-effort; robust fallbacks)*
**Optional Inputs (observed):** `market_data` (direct injection, compact current-bar or multi-TF)
**Side-effects:** publishes only theme-prefixed keys to SmartInfoBus; background health thread; performance metrics; KMeans (MiniBatch) fitting with circuit-breaker; writes `theme_model_quality` and `theme_detector_health`
**Voting?** no (mixins include voting/trading/state helpers, but not a voter)
**Explainable?** yes (`thesis_required=True`; returns `_thesis` + `thesis`)
**Timeout/Priority:** not explicitly time-boxed in decorator; background monitor tick \~30s
**Key Methods:** `process`, `_extract_market_data`, `_process_theme_detection`, `_extract_comprehensive_features`, `_fit_model_safe`, `_detect_current_theme`, `_calculate_theme_confidence`, `_generate_theme_thesis`, `_update_theme_smart_bus`, `propose_action`, `calculate_confidence`, `get_state/set_state`, `get_health_status`
**State:** current theme id, one-hot strength vector, confidence, histories (strength/momentum/theme ids), feature buffer for fitting, clustering quality, feature stability, ML circuit breaker, success/failure counters
**Circuit breakers:** ML training breaker (OPEN after N failures; reset after 300s idle) + soft “circuit\_breaker\_failures” metric
**Consumers (TBD):** regime/portfolio overlay, UI dashboards, risk posture
**Input owners (likely):** market data provider / orchestrator writing `historical_prices` or `multi_timeframe_data` and `macro_data`
**Conflicts/Duplicates:** avoids collisions—only writes theme-namespaced keys. It *includes* `market_data`/`price_data`/`technical_indicators` inside its **returned payload** (for dashboards), but **does not** publish those keys to the bus.

---

## RAW CAPTURE — MarketThemeDetector

**Class:** `MarketThemeDetector(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):**

* `name="MarketThemeDetector"`, `version="3.1.0"`, `category="market"`
* **provides:** `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`, `theme_analysis`, `theme_detection`, `theme_detector_status`, `theme_detector_health`, `theme_model_quality`
* **requires:** `market_data`, `price_data`, `technical_indicators`, `historical_prices`, `multi_timeframe_data`, `macro_data`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config (`ThemeDetectorConfig`)

* `n_themes=4`, `window=100`, `batch_size>=64` (auto: `max(64, n_themes*16)`), `feature_lookback=500`
* `instruments`: defaults to `["XAU/USD", "EUR/USD"]`
* ML: `max_iter=100`, `convergence_threshold=0.001`, `clustering_quality_threshold=0.30`
* Perf: `max_processing_time_ms=200`, `circuit_breaker_threshold=3`
* Features: `use_macro=True`, `timeframes=("H1","H4","D1")`

### 2) Lifecycle & systems

* Initializes SmartInfoBus, rotating logger, `ErrorPinpointer`, `EnglishExplainer`, `PerformanceTracker`.
* ML stack: `StandardScaler` + `MiniBatchKMeans(n_clusters=n_themes, batch_size, max_iter, n_init=10)`.
* Background monitoring **thread** (daemon) every \~30s: updates health metrics & resets ML breaker after 300s.
* Publishes an initial `theme_detector_status`.

### 3) Data extraction (robust, multi-path)

Order of precedence:

1. `historical_prices` **or** `multi_timeframe_data` (multi-TF dict) → used directly.
2. `market_data` (compact current-bar dict) → shaped into `{instrument: {H4: {...}}}` with arrays and `current_bar`.
3. Individually keyed: `market_data_{instrument}_{timeframe}` for configured instruments×TFs.
4. Direct `inputs["market_data"]` if passed to `process`.
5. **Last known** cached market data (warn).
6. **Synthetic** generator (deterministic-ish) for configured instruments×TFs.

### 4) Feature engineering

For **each** instrument×timeframe, appends 7 stats (uses closing prices):

* `vol` (stdev of last 20 returns), `mom` (mean last 5 returns), `hurst` (safe est on last 50),
  `wave` (detail energy on last 30 via pywt db4), `trend` (SMA10 vs SMA30),
  `roll_10` (last/\[-10] − 1), `bars_available` (count).
  If `use_macro`: appends **3 macro** features `[vix, yield_curve, cpi]`, scaled via a separate scaler (seeded with `[20, 0.5, 3]`).
* Feature vector is padded/trimmed to a **fixed length**:
  `expected = len(instruments) * len(timeframes) * 7 + (3 if use_macro else 0)`.

### 5) Model fitting & readiness

* A rolling **fit buffer** (`deque`, maxlen 2000) collects recent feature vectors.
* `_should_fit_model()` → **fit** when `len(buffer) >= batch_size` **and** `(ml_fit_count % 10 == 0)`.
  (i.e., fits at counts 0, 10, 20… to throttle.)
* Fit pipeline: `X` → scale → `MiniBatchKMeans.fit(X_scaled)`; increments `ml_fit_count`.
  Computes **quality** = `1 / (1 + inertia/n_samples)` (clamped 0..1).
  Publishes `theme_model_quality` to the bus.
* **Readiness**: cluster centers exist **and** `quality > clustering_quality_threshold` (default 0.30).

### 6) Detection & scores

* If **ready**:

  * `theme_id = argmin_k distance_k(kmeans.transform(fs))` where `fs=scaled(features)`.
  * `theme_strength = 1 / (1 + min_distance)` (→ 0..1).
  * `theme_confidence` \~ **separation**: `(second_smallest − min) / second_smallest` (0..1).
* Else (cold start): defaults → `theme_id=0`, `strength≈0.30`, `confidence≈0.10`.
* Tracks histories: `_theme_vec` one-hot with strength, `_theme_momentum` (last strengths), `_theme_history`.

**Stability & transitions**

* `theme_stability`: `1 − std(last 10 strengths)` (clipped 0..1; fallback 0.5 if <5 points).
* `transition_probability`: from last 3 strengths’ **momentum**: `prob = clip(-avg_diff + 0.10, 0..1)`.

### 7) Outputs (contract & payload)

The **returned dict** always includes:

```yaml
market_theme: int                         # 0..n_themes-1
theme_strength: float                     # 0..1
theme_confidence: float                   # 0..1
theme_transition: float                   # alias of transition_probability
theme_detection:                          # compact blob for downstreams
  theme: int
  strength: float
  confidence: float
  stability: float
  transition_probability: float
  timestamp: ISO8601
theme_analysis:
  theme_vector: float[]                   # one-hot style strengths
  recent_transitions: int
  last_update: ISO8601
  ml_quality: float
  data_quality: float
# convenience mirrors (for dashboards; not bus-published):
market_data: dict                         # upstream snapshot or derived
price_data: dict                          # derived OHLC per instrument
technical_indicators: dict                # passthrough if present
market_features: float[]                  # last feature vector
thesis: string
_thesis: string
processing_success: bool
```

### 8) SmartInfoBus writes (side-effects)

* `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`
* `theme_detection` (compact payload)
* `theme_analysis` (snapshot incl. thesis)
* `theme_detector_status` (init) and `theme_detector_health` (periodic)
* `theme_model_quality` (on each successful fit)
* **Does not** publish `market_data`/`price_data`/`technical_indicators` to avoid namespace clashes.

### 9) Thesis generation

* Named themes:
  `0 Risk-Off Defensive`, `1 Growth Momentum`, `2 Volatility Spike`, `3 Range-Bound Consolidation` (defaults to `Theme {id}` for others).
* Human summary includes: Strength/Confidence/Stability (with labels), instruments analyzed count, clustering quality, feature stability, transition-risk notice, fit count & buffer size, timestamp.

### 10) Health, errors & breakers

* **ML circuit breaker:** tracks failures during fit; `OPEN` after `threshold` (default 3); auto-reset to `CLOSED` after 300s.
* General error path returns **safe fallback** with `theme_detection` filled and explanatory thesis; logs via `ErrorPinpointer` + `EnglishExplainer`.
* Background health snapshot publishes `theme_detector_health`: success rate, avg processing time, breaker state, quality, data extraction rate, timestamp.

### 11) Actions & confidence (helpers)

* `propose_action()` maps theme → action/risk:

  * **0 Risk-Off:** `reduce_exposure` (if conf ≥0.5) else `monitor` (risk: medium)
  * **1 Momentum:** `buy_moderate` or `buy_aggressive` (risk: medium; scaled by confidence)
  * **2 Vol Spike:** `defensive` (risk: high)
  * **3 Range-Bound:** `range_trade` (if conf ≥0.6) else `monitor` (risk: low)
* `action_confidence` = blend of conf/strength/stability/quality (bounded 0..1).
* `calculate_confidence()` mixes: conf, clustering quality, stability, ML breaker health, data extraction success; adjusts for aggressive calls & alignment with range-trade.

### 12) State I/O

* `get_state()` returns current theme, vector, transitions count, quality, fit count, success/failure counts, config subset, last update.
* `set_state()` safely restores those and logs success.

---

## Data shapes (quick reference)

**Input expectations (any subset):**

```yaml
historical_prices | multi_timeframe_data:
  <instrument>:
    <timeframe>:
      open/high/low/close: float[]
      volume: int[]
      current_bar: {open, high, low, close, volume}
      timeframe: str
      bars_available: int

market_data (compact):
  <instrument>: {open, high, low, close, volume, bid?, ask?}

macro_data:
  vix: float
  yield_curve: float
  cpi: float
```

**Outputs (returned payload):**

```yaml
market_theme: int
theme_strength: float
theme_confidence: float
theme_transition: float
theme_detection: {...}
theme_analysis: {...}
market_data: dict
price_data: dict
technical_indicators: dict
market_features: float[]
thesis: string
_thesis: string
```

**Bus keys written:**
`market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`, `theme_detection`, `theme_analysis`, `theme_detector_status`, `theme_detector_health`, `theme_model_quality`.

---

all set. send the next file whenever you’re ready.

sweet — here’s the clean capture + ledger entry for **RegimePerformanceMatrix** in the same format.

---

## 📒 Master ledger (append)

**Module:** RegimePerformanceMatrix
**File:** `modules/market/regime_performance_matrix.py`
**Category:** market
**Provides:**
`regime_accuracy`, `regime_prediction`, `market_state`, `performance_metrics`, `stress_test_results`, `backtesting_data`, `recent_trades`, `trading_signals`, `regime_analysis`, `regime_data`, `regime_matrix_analysis`, `regime_matrix_health`, `regime_matrix_status`, `regime_performance`
**Requires (best-effort):**
`market_regime`, `market_data`, `liquidity_score`, `recent_trades`, `volatility_data`, `pnl_data`
**Optional / inferred:** derives volatility from `market_data` multi-TF when `volatility_data` is absent
**Side-effects (SmartInfoBus writes):**
`regime_performance`, `regime_accuracy`, `regime_prediction`, `regime_data`, `regime_analysis`, `regime_matrix_analysis`, `market_state`, `performance_metrics`, `backtesting_data`, `recent_trades` *(passthrough)*, `trading_signals`, `stress_test_results`, `regime_matrix_health`, `regime_matrix_status`
**Single-writer discipline:** does **not** publish `market_regime` (reads it only).
**Explainable:** yes (`thesis_required=True`; returns `thesis` + `_thesis`)
**Background:** daemon monitor thread \~30s (health + per-regime accuracy refresh)
**Perf tracking:** `performance_tracker.record_metric("RegimePerformanceMatrix", "matrix_processing", …)`
**Circuit breaker:** soft counter (`circuit_breaker_failures`), no hard OPEN/CLOSED gate

---

## RAW CAPTURE — RegimePerformanceMatrix

**Class:** `RegimePerformanceMatrix(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="RegimePerformanceMatrix"`, `version="3.1.0"`, with *provides*/*requires* as above.

### 1) Config (`RegimeMatrixConfig`)

* `n_regimes=3`, `decay_factor=0.95`, `vol_history_size=500`, `performance_window=100`, `regime_sensitivity=1.0`
* Perf thresholds: `max_processing_time_ms=150`, `circuit_breaker_threshold=3`, `accuracy_threshold=0.60`
* Instruments for vol fallback: `("XAU/USD", "EUR/USD")`

### 2) State & structures

* `matrix`: **n×n** float32 (exponentially-decayed P\&L by *predicted*→*true* regime)
* `volatility_regimes`: 3 anchors initial `[0.1, 0.3, 0.5]` → adapt from history percentiles
* Pointers: `_current_regime`, `_predicted_regime`, `last_volatility`, `last_liquidity`
* Histories: `vol_history`, `_performance_history`, `_regime_history`, `_predicted_regime_history`, `_true_regime_history`
* Per-regime: `_regime_accuracy_scores[n]`, `_regime_pnl_tracking[i]->deque`, `_regime_transitions` map `"i->j"`→stats
* Characteristics per regime: `avg_volatility`, `avg_pnl`, `count`, `accuracy`, `stability_score`
* Monitoring counters: `processing_times`, `success_count`, `failure_count`, `circuit_breaker_failures`

### 3) Data extraction

`_extract_performance_data()` builds:

* **Predicted regime** from `market_regime` (accepts str/int). Mapping for strings:
  `{"trending": 0, "volatile": 1, "ranging": 2}` *(anything else → 0)*
* **Volatility**: from `volatility_data` or `_calculate_volatility_fallback()` using `market_data` **multi-TF** `{close:[...]}` (std of last ≤20 returns).
* **PnL**: `pnl_data` numeric, else sum of `recent_trades[].pnl` (if list).
* **Liquidity**: `liquidity_score` (default 1.0).
  Returns a compact dict with timestamp + source.

> ⚠️ Heads-up: regime string mapping (`trending/volatile/ranging`) may not align with other modules that use `noise/volatile/trending`. Unknown strings → 0.

### 4) Core logic

`_process_regime_matrix(performance_data)`:

* **True regime** = nearest anchor in `volatility_regimes` (Euclidean distance).
  Anchors update when `vol_history > 50` using percentiles: `[33rd, 66th, 90th]`.
* Update histories (`pred`, `true`, `vol`, `pnl`), append to matrix cell `(pred,true)` via **exponential decay**:
  `M[i,j] = M[i,j]*decay + pnl*(1-decay)`
* If regime changed: `_handle_regime_transition(old,new,vol,pnl)` increments `"old->new"` counts and rolling avgs; logs.
* Update `_current_regime`, `_predicted_regime`, `last_volatility`; refresh regime characteristics (rolling averages).
* Metrics:

  * `overall_accuracy`: proportion of periods with `pred==true`
  * `regime_accuracy(r)`: accuracy **conditional** on true==r
  * `avg_performance`: mean of `_performance_history`
  * `volatility_trend`: slope sign of polyfit over last 10 vols (`increasing/decreasing/stable`)

### 5) Outputs (returned payload)

`_format_declared_outputs(...)` guarantees all *provides* and includes extras:

* **Declared provides:**

  * `regime_accuracy` (overall)
  * `regime_prediction` `{predicted, actual, correct}`
  * `market_state` `{regime, volatility, trend}`
  * `performance_metrics` `{overall_accuracy, regime_accuracy, avg_performance, processing_success}`
  * `stress_test_results` (last results dict)
  * `backtesting_data` `{window, volatility_history[-50:], timestamp}`
  * `recent_trades` (from bus or injected)
  * `trading_signals` `{signal, confidence, timestamp}`
  * `regime_analysis` (rich blob incl. `last_update`)
  * `regime_data` `{matrix, characteristics, volatility_regimes}`
  * `regime_performance` `{matrix, current_regime, predicted_regime, avg_performance}`
* **Also returns (convenience, not in provides):** `market_regime` (int), `thesis`, `_thesis`
* **Signal rule:** if `overall_accuracy > max(0.6, accuracy_threshold)` →
  `signal = "trade"` when `current==predicted`, else `"reduce_exposure"`; otherwise `"hold"`.
  `confidence = min(1, overall_accuracy + (1 - current_volatility)*0.2)`

### 6) SmartInfoBus writes

* Performance & analytics: `regime_performance`, `regime_accuracy`, `regime_prediction`, `regime_data`, `regime_analysis`, `regime_matrix_analysis`
* State & metrics: `market_state`, `performance_metrics`, `backtesting_data`, `stress_test_results`
* Passthrough: `recent_trades` (reads then re-sets)
* Signals: `trading_signals`
* Health & status (via monitor / init): `regime_matrix_health`, `regime_matrix_status`
* **Note:** explicitly avoids writing `market_regime` (to preserve single-writer policy).

### 7) Thesis

`_generate_matrix_thesis(...)` → compact narrative with:

* True vs Predicted regime (named: 0 Low Vol, 1 Medium Vol, 2 High Vol), prediction status, overall accuracy
* Average performance, current volatility, trend, decay factor
* Per-regime characteristics (obs, avg vol, avg PnL, accuracy)
* Accuracy tier blurb, plus quick stats: transition count, performance window fill, vol history fill

### 8) Monitoring, fallbacks & errors

* **Monitor thread (30s):** `_update_health_metrics()` + `_update_regime_accuracy()`; publishes `regime_matrix_health`.
* **No data:** returns safe matrix & metrics with thesis; logs warning.
* **Errors:** logs via `ErrorPinpointer`/`EnglishExplainer`, increments failure counters, then falls back.

### 9) Actions & confidence (helpers)

* `propose_action()` (uses current regime, predicted, vol, overall accuracy):

  * Regime 0 (Low vol): `"buy"` if `pred==curr` and acc>0.7 else `"hold"` (risk low)
  * Regime 1 (Med vol): `"trade"` if acc>0.6 else `"reduce_exposure"` (risk medium)
  * Regime 2 (High vol): `"defensive"` if vol>0.3 else `"cautious_trade"` (risk high)
  * `regime_confidence = min(1, overall_accuracy + (1 - vol)*0.3)`
* `calculate_confidence(action)` weighted blend:
  `0.40*overall_accuracy + 0.25*regime_stability + 0.15*data_quality + 0.10*perf_consistency + 0.10*matrix_coverage`, with action-specific tweaks.

### 10) State I/O

* `get_state()` returns matrix, pointers, anchors, characteristics, transitions, accuracy scores, counters, config slice.
* `set_state()` restores shapes safely; logs success.

---

## Quick integration notes / gotchas

* **Regime label mapping:** Reads `market_regime` strings as `trending→0`, `volatile→1`, `ranging→2`. If your upstream uses `noise` instead of `ranging`, this will default to **0**. Consider normalizing upstream values or extend the mapping.
* **recent\_trades writer:** This module **re-publishes** `recent_trades`. If other modules write to the same key, coordinate single-writer or designate a canonical owner.
* **Volatility scale assumptions:** Signal confidence uses `(1 - volatility)`. Works best when vol ∈ \~\[0,1]; if your vol is larger, consider scaling before publishing to the bus.
* **Stress scenarios:** Predefined in `_initialize_stress_testing()` but only surfaced via `stress_test_results`; no automatic execution path shown here.

---

all set. want me to do the next file the same way?


sweet — here’s the clean capture + ledger entry for **TimeAwareRiskScaling** in the same format.

---

## 📒 Master ledger (append)

**Module:** TimeAwareRiskScaling
**File:** `modules/market/time_aware_risk_scaling.py`
**Category:** risk
**Provides:**
`risk_scaling_factor`, `session_risk`, `volatility_adjustment`, `market_conditions`, `time_risk_analysis`, `time_risk_status`, `time_risk_health`
**Requires (best-effort):**
`timestamp`, `market_data`, `risk_data`, `volatility_data`
**Side-effects (SmartInfoBus writes):**
`risk_scaling_factor`, `session_risk`, `volatility_adjustment`, `market_conditions`, `time_risk_analysis`, `time_risk_status` *(on init)*, `time_risk_health` *(via monitor)*
**Explainable:** yes (`thesis_required=True`; returns `thesis` + `_thesis`)
**Background:** daemon monitor thread \~30s (health + session pattern + threshold checks)
**Perf tracking:** `performance_tracker.record_metric("TimeAwareRiskScaling", "risk_scaling", …)`
**Circuit breaker:** soft counter only (`circuit_breaker_threshold` drives logging/alerts; no hard gate)

---

## RAW CAPTURE — TimeAwareRiskScaling

**Class:** `TimeAwareRiskScaling(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="TimeAwareRiskScaling"`, `version="3.1.0"` with provides/requires as above.

### 1) Config (`TimeAwareRiskConfig`)

* **Session ends (UTC):** `asian_end=8`, `euro_end=16`, `us_end=22`
* **Risk scaling:** `base_factor=1.0`, `decay_factor=0.9`, `vol_window=100`, `session_memory=24`
* **Session multipliers:** asian `1.2`, european `1.0`, US `1.1`, closed `0.5`
* **Thresholds:** `max_processing_time_ms=100`, `circuit_breaker_threshold=3`, `risk_threshold_high=0.8`, `risk_threshold_critical=0.95`
* **Volatility fallback instruments:** `("XAU/USD", "EUR/USD")`

### 2) State & structures

* **Per-hour arrays (length 24):** `vol_profile`, `risk_profile`, `_hourly_risk_scores`
* **Session tracking:** `_current_session`, `_session_changes`, `_session_transitions` (deque)
* **Risk tracking:** `_volatility_history` (size `vol_window`), `_factor_history`, `_risk_events`
* **Session stats:** `_session_performance[session] = {count, total_factor, avg_volatility, risk_events, success_rate, last_update}`
* **Dynamic multipliers:** `_session_risk_multipliers` (auto-tuned by success rate after ≥10 obs)

### 3) Data extraction

* `_extract_time_data()`

  * **timestamp:** from bus `timestamp` (str/ts/dt) else now; **hour** = UTC hour
  * **volatility:** `_extract_volatility_data()` uses `volatility_data` numeric; else **expects** `market_data[instrument]["close"]` (simple series, not multi-TF) to compute std of recent returns; else falls back to `current_volatility` or `0.01`
  * **session:** `_get_session(hour)` via config cutoffs
  * pulls passthrough `market_data`, `risk_data`

### 4) Core logic

* `_process_time_aware_scaling(time_data)`

  * Session transition bookkeeping (+ logging) and adaptive multiplier tweak
  * **Base factor:** `base_factor` decayed by recent vol ratio; mild hourly normalization via `vol_profile`
  * **Vol adjustment:** z-score on vol history → returns {1.5, 1.2, 1.0, 0.85, 0.7} bands
  * **Session multiplier:** from config, auto-adjusted by performance (`success_rate>0.8 → ×0.95`, `<0.6 → ×1.05`, clipped 0.3..2.0)
  * **Final scaling factor:** `clip(base*volAdj*sessionMult, 0.1, 5.0)`
  * **Risk level (0..1):** `0.4*factor_risk(scaling/2)` + `0.4*vol_risk(percentile of vol history)` + `0.2*session_risk_map` (asian .3, eur .2, us .25, closed .1)
  * Updates per-hour profiles; computes `risk_trend` (slope on last 5 factors), `volatility_trend` (last 10 vols), `volatility_regime` (percentile bands), `session_efficiency` (success\_rate penalized by risk\_events)

### 5) Outputs (returned payload)

* **Declared provides:**

  * `risk_scaling_factor` (float)
  * `session_risk` `{current_session, risk_level, session_multiplier, hour}`
  * `volatility_adjustment` `{adjustment_factor, current_volatility, volatility_regime, volatility_trend}`
  * `market_conditions` `{session, hour, volatility_regime, risk_trend}`
  * `time_risk_analysis` rich blob (risk/vol trends, hourly patterns, transitions, success flag, last\_update)
  * `time_risk_status` `{status, current_session, hour, scaling_factor, risk_level, volatility, last_update}`
  * `time_risk_health` `{success_rate, avg_processing_time_ms, circuit_breaker_failures, current_risk_level, session_transitions, risk_events, last_update}`
* **Also returns (convenience):** `volatility_data` (float), `risk_data` (compact dict), `thesis`, `_thesis`

### 6) SmartInfoBus writes

* On **process**: `risk_scaling_factor`, `session_risk`, `volatility_adjustment`, `market_conditions`, `time_risk_analysis`
* On **init**: `time_risk_status`
* On **monitor** (30s): `time_risk_health`
* Perf metric recorded each run

### 7) Thesis

* `_generate_risk_thesis(...)` → narrative with session name, hour (UTC), scaling factor, risk level, current vol + regime, component breakdown, per-session guidance for **Gold (XAU/USD)** & **EUR/USD**, regime warnings, threshold notices, and session performance summary.

### 8) Monitoring, thresholds & fallbacks

* **Monitor loop (30s):** updates health, analyzes session patterns every ≥300s (`_hourly_risk_scores`), checks thresholds → pushes `_risk_events` & logs `[ALERT]` if `current_risk_level` exceeds `risk_threshold_high/critical`
* **No data / errors:** safe fallback via `_format_declared_outputs` with thesis; errors annotated via `ErrorPinpointer`/`EnglishExplainer`

### 9) Actions & confidence

* `propose_action()` calls `process()` and maps `risk_level` → `{reduce_exposure, moderate_caution, maintain_current, increase_exposure}` with `magnitude` and `confidence = min(0.9, scaling_factor/2)`
* `calculate_confidence(action)` = blend of base (history aware), scaling proximity to 1.0, risk bucket; +10% if session known

### 10) State I/O

* `get_state()` returns session markers, profiles, performance, multipliers, counters, and key config fields
* `set_state()` restores safely; logs success
* `stop_monitoring()` flips flag (daemon exits naturally)

---

## Quick integration notes / gotchas

* **Market data shape for vol fallback:** expects `market_data[INSTR]["close"]` directly under the instrument (not the multi-TF `{TF:{close:[…]}}` shape other modules use). If you publish multi-TF, consider adding a simple snapshot alongside for this module.
* **UTC assumption:** session mapping uses UTC hour; ensure your `timestamp` is UTC or adjust ends accordingly.
* **Double processing in `propose_action()`:** it calls `process()` again, which re-writes bus keys. If you chain actions frequently, consider passing cached results to avoid redundant writes.
* **Scaling → confidence coupling:** confidence caps at `0.9` and scales with factor/2; very large factors will not exceed that cap by design.
* **Threshold alerts:** only log & enqueue in `_risk_events`; there’s no external notifier inside this module. If you need paging, subscribe downstream to `time_risk_health` or `time_risk_analysis`.

---

all set. want me to capture the next file too?


sweet — here’s the clean capture + ledger entry for **HistoricalReplayAnalyzer** in the same format.

---

## 📒 Master ledger (append)

**Module:** HistoricalReplayAnalyzer
**File:** `modules/memory/historical_replay_analyzer.py`
**Category:** memory
**Provides:** `replay_sequences`, `pattern_analysis`, `sequence_quality`, `learning_progress`
**Requires (best-effort):** `trades`, `actions`, `market_data`, `episode_data`
**Side-effects (SmartInfoBus writes):**

* On init: `replay_sequences` (initial status)
* On each process: `replay_sequences`, `pattern_analysis`, `sequence_quality`, `learning_progress`
* Health/metrics: records performance metrics; optional `record_module_timing` / failure notices
  **Explainable:** yes (`thesis_required=True`; returns `thesis` via `_thesis` in the response and used in bus writes)
  **Background:** lightweight monitor thread (\~30s) updating health & “effective patterns” logs
  **Perf tracking:** `performance_tracker.record_metric("HistoricalReplayAnalyzer","analysis_cycle",…)`
  **Circuit breaker:** simple local CB (`OPEN` after N failures; auto-closes on success)

---

## RAW CAPTURE — HistoricalReplayAnalyzer

**Class:** `HistoricalReplayAnalyzer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="HistoricalReplayAnalyzer"`, `version="3.1.0"`, `category="memory"`

### 1) Config (`ReplayConfig`)

* Cadence/logic: `interval` (episodes between auto-replays), `bonus`, `sequence_len`, `profit_threshold`, `pattern_sensitivity`, `replay_decay` (bonus *decays* as `base ** hours_since_event`)
* Performance/safety: `max_processing_time_ms`, `circuit_breaker_threshold`, `min_sequence_quality`
* Analysis limits: `max_sequences`, `lookback_episodes`, `pattern_confidence_threshold`
* Integration: optional `namespace` for namespaced bus keys

### 2) Systems & validators

* Registers SmartInfoBus validators for all 4 provided keys (also namespaced variants if `namespace` set) ensuring stable shapes.
* Optional `InfoBusManager.register_module_capabilities` call (best-effort).
* Subscribes to generic `performance_warning` (no-op hook).

### 3) State

* Episode ring buffers: `episode_buffer` (size = `lookback_episodes`), `_episode_count`
* Working sets: `current_sequence`, `profitable_sequences` (Top-K pruning to `max_sequences`)
* Pattern store: `sequence_patterns[pattern] = {count,total_pnl,avg_pnl,last_seen,confidence}`
* Quality & learning: `_sequence_quality_scores`, `_pattern_evolution`, `_learning_curve`, `_analysis_performance` counters
* Bests: `best_sequence_pnl`, `replay_bonus`
* Circuit breaker + health fields

### 4) Data extraction

`_extract_sequence_data()` reads (optionally namespaced):

* `trades`, `actions`, `market_data`, `episode_data` from the bus; normalizes `current_action` (arraylike → list).
* Adds ISO `timestamp` and `episode_completed` flag (from `inputs`).

### 5) Core process

`process()` flow:

1. Extract sequence data; otherwise return cached **fallback** (with all provides + `_thesis`) and still publish to bus.
2. `_process_trading_sequence()`

   * Append current step `{action,timestamp,market_context}`; trim to `sequence_len`.
   * Compute **sequence\_quality**:

     * Action magnitude variance → **consistency\_score** (lower variance = better)
     * Temporal spacing std → **temporal\_score**
     * Market context (volatility dispersion) → **context\_score** (prefers moderate vol)
   * Increment `sequences_analyzed`.
3. If `episode_completed`: `_analyze_episode_patterns()`

   * Buffer episode `{sequence,pnl,timestamp,market_conditions}`; ++episode count.
   * If `pnl > profit_threshold`: `_analyze_profitable_sequence()` (store entry, update best, prune Top-K).
   * Extract discrete **pattern signature** from the sign of the **first action component** per step: `L`/`S`/`H` string; update pattern stats + confidence.
4. `_generate_replay_recommendations()`

   * If `interval>0` and episode count is a multiple → compute **replay\_bonus** = `bonus * (pnl/100) * (replay_decay ** hours_since_best)`.
   * Pick **best\_pattern** via `avg_pnl * confidence`.
5. Compose **thesis** with processed counts, bests, whether replay triggered, average recent quality, and learning trend.
6. Publish all four provided payloads to SmartInfoBus; record perf metrics; return contract payload + `_thesis`.

### 6) Outputs (return payload & bus shapes)

* **`replay_sequences`**: `{total_sequences, best_sequence_pnl, replay_bonus, sequences_analyzed}`
* **`pattern_analysis`**: `{total_patterns, profitable_patterns, best_pattern, pattern_confidence_avg}`
* **`sequence_quality`**: `{current_quality, average_quality, quality_trend, episodes_processed}`
* **`learning_progress`**: `{profitable_sequences_count, total_episodes, success_rate, learning_acceleration}`
* Plus `_thesis` (string). Fallback/error paths always include these keys.

### 7) Monitoring & health

* Every \~30s:

  * `_update_replay_health()` sets status `healthy`/`warning` using pattern discovery rate and recent avg quality vs `min_sequence_quality`.
  * `_analyze_pattern_effectiveness()` logs when `count≥5` and `avg_pnl*confidence > 10`.

### 8) Error & fallback behavior

* Local CB increments on error; `OPEN` after threshold; **auto-closes** on next success.
* `_handle_no_data_fallback()` and `_create_fallback_response(reason)` both return fully-formed provides + `_thesis`, preserving last known state.

### 9) Legacy hooks

* `propose_action()` returns a neutral stub (no trading actions).
* `calculate_confidence(action)` blends recent sequence quality with pattern confidence of the **augmented** sequence including the proposed action.

---

## Quick integration notes / gotchas

* **Namespaces:** If you set `config.namespace = "ns"`, the module *both* reads and writes namespaced keys (e.g., `ns:replay_sequences`) **and** registers validators for them. Make sure your producers/consumers use the same namespace.
* **Action shape:** Pattern extraction only looks at the **first** action component per step; if your policy vector has a different semantics, consider adjusting or mapping before feeding.
* **Learning acceleration clock:** uses a cutoff of `recent_window_episodes * 60s` from *now* (episode-count–based time proxy). If your episodes aren’t \~1 minute long on average, this metric may be skewed; swap to wall-clock or explicit episode durations if needed.
* **Replay cadence:** Auto-replay triggers only when `interval>0` **and** episode\_count % interval == 0 **and** there’s at least one profitable sequence.
* **Vol context in quality:** If you want the context factor to be meaningful, ensure your `market_data` propagated into each step includes a `volatility` value.

---

all set. want me to capture another file?

nice — here’s the clean capture + ledger entry for **MemoryBudgetOptimizer** in the same format.

---

## 📒 Master ledger (append)

**Module:** MemoryBudgetOptimizer
**File:** `modules/memory/memory_budget_optimizer.py`
**Category:** memory
**Version:** 3.0.1
**Provides:** `memory_allocation`, `budget_optimization`, `memory_efficiency`, `allocation_strategy`
**Requires (hard):** `memory_usage`
**Soft/optional reads:** `trades`, `mistakes` *(or `loss_prevention.{recent_losses|loss_events}`)*, `playbook_entries` *(or `pattern_memory.pattern_effectiveness`)*, `performance_metrics`
**Side-effects (SmartInfoBus writes):**

* On init: `memory_allocation` (initial sizes)
* Each `process()`: `memory_allocation`, `memory_efficiency`, `budget_optimization`, `allocation_strategy`
  **Explainable:** yes (`thesis_required=True`; `_thesis` returned and used in bus writes)
  **Background:** singleton monitor thread for the class (\~30s loop) updating health & utilization warnings
  **Perf tracking:** `performance_tracker.record_metric('MemoryBudgetOptimizer','optimization_cycle',…)`
  **Circuit breaker:** local dict; `OPEN` after threshold failures; auto-resets on success

---

## RAW CAPTURE — MemoryBudgetOptimizer

**Class:** `MemoryBudgetOptimizer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="MemoryBudgetOptimizer"`, `version="3.0.1"`, `category="memory"`

### 1) Config (`MemoryBudgetConfig`)

* Budgets: `max_trades`, `max_mistakes`, `max_plays`, `min_size` (per-type floor)
* Perf/safety: `max_processing_time_ms`, `circuit_breaker_threshold`, `optimization_interval` (episodes between rebalances)
* Allocation controls: `rebalance_sensitivity`, `efficiency_weight`, `recency_weight`, `utilization_target`
* Hidden knobs via `getattr`: `warn_min_interval_sec` (default 180s), `warn_on_state_change_only` (default True)

### 2) Systems & state

* **Shared logger** (class-level, lock-protected) to avoid spammy init lines
* Local CB & health status
* **Memory performance buckets**: `trades`, `mistakes`, `plays` each track `size`, `hits`, `profit`, `recent_hits`, `efficiency`
* Analytics: `_optimization_history`, `_efficiency_trends`, `_allocation_changes`, `_memory_utilization_history`, `_resource_waste_tracking`
* Flags for data gating in monitor: `_has_seen_data`, `_no_data_notice_emitted`, `_util_state`, `_last_warn`

### 3) Data extraction

`_extract_memory_data()` pulls (best-effort):

* `trades` (list of dicts with `pnl`)
* `mistakes` (list with `cost`), else derives from `loss_prevention.{recent_losses|loss_events}`
* `playbook_entries` (list with `value`), else derives from `pattern_memory.pattern_effectiveness`
* **Required:** `memory_usage` (shape-agnostic; not strictly used in scoring but part of contract)
* `performance_metrics` (optional)

### 4) Core process

1. Update per-bucket **hits/profit** from last N items; mark `_has_seen_data` once anything increments.
2. Compute **utilization metrics** per bucket: `efficiency = profit/hits`, `utilization = hits/size`, `recent_hit_rate`.
3. Track efficiency trends, compute overall **optimality\_score** from recent optimization history.
4. If `_should_optimize()` → compute **efficiency\_scores** (profit-per-hit, utilization factor vs target, recency factor), convert to **optimal\_allocation** under total budget = `max_trades + max_mistakes + max_plays`; apply significant changes (≥ 10% of `min_size`).
5. Build **thesis** (avg efficiency, best performer, optimality, recent changes; note if optimization ran).
6. Publish provides to the bus and record perf metrics.
7. Fallback/error paths always return all provides + `_thesis`.

### 5) Monitoring & health

* Every \~30s:

  * `_update_memory_health()` classifies `healthy|warning|critical` from avg efficiency (total\_profit / total\_hits).
  * `_check_allocation_efficiency()` issues **rate-limited warnings** on **LOW** (<10%) or **HIGH** (>95%) utilization per bucket; waits until first data arrives.

### 6) Outputs (return payload + bus shapes)

* **`memory_allocation`**: `{"trades": int, "mistakes": int, "plays": int}` (current sizes)
* **`memory_efficiency`** *(utilization\_metrics)* per bucket: `{"efficiency": float, "utilization": float, "recent_hit_rate": float, "total_hits": int, "total_profit": float, "size": int}`
* **`budget_optimization`**: `{"optimality_score": float, "total_profit": float, "optimization_count": int, "last_optimization": float}`
* **`allocation_strategy`**: `{"allocation_method": "efficiency_based", "rebalance_frequency": int, "efficiency_weight": float, "recent_changes": int}`
* Plus extras in the return payload: `memory_performance`, `efficiency_trends`, and `_thesis`.

---

## Quick integration notes / gotchas

* **Runs on first cycle:** `_should_optimize()` returns `True` when `optimization_count % optimization_interval == 0`; with `optimization_count=0`, the first `process()` triggers an optimization by design.
* **`total_profit` is never updated:** the code tracks per-bucket `profit` but does not accumulate into `self.total_profit` (used in thesis and bus). Consider summing the three bucket profits each cycle (or updating it when hits change).
* **`memory_usage` is required but not consumed** in scoring; safe but FYI if upstream expects it to steer allocation.
* **Change threshold:** `_apply_allocation_changes()` only applies when `|change| >= 10% of min_size` (with default `min_size=50`, threshold is 5). Tiny reallocations won’t be applied or logged.
* **Budget is fixed to genome maxima:** total budget for realloc = `max_trades + max_mistakes + max_plays`. If you mutate sizes externally, they’ll be pulled back toward this total on next optimize.
* **Warning flood control:** utilization warnings are gated both by **state change** and **min interval** (default 180s) and suppressed until real data is seen.
* **Threading:** one class-wide monitor thread; `stop_monitoring()` correctly flips both instance and class flags.

---

want me to capture another file the same way?
nice — here’s the clean capture + ledger entry for **MemoryCompressor** in the same format.

---

## 📒 Master ledger (append)

**Module:** MemoryCompressor
**File:** `modules/memory/memory_compressor.py`
**Category:** memory
**Version:** 3.0.0
**Provides:**
`intuition_vector`, `compressed_patterns`, `memory_compression`, `feature_importance`
**Requires (declared):**
`trades`, `features`, `market_context`, `episode_data`
**Explainable:** yes (`thesis_required=True`; `_thesis` returned & used in bus writes)
**Priority / timing hints:** none declared (no `timeout_ms` / `priority`)
**Concurrency / background:** per-instance monitoring thread (`_start_monitoring` runs every 30s)
**Perf tracking:** `performance_tracker.record_metric("MemoryCompressor","compression_cycle", …)` on success/failure
**Circuit breaker:** `circuit_breaker_threshold=3`; on threshold → state `OPEN`, fallback payload used

**Side-effects (SmartInfoBus writes):**

* On init:

  * `memory_compression` (initial status: counts, intuition vector, efficiency)
* Each `process()` (comprehensive update):

  * `intuition_vector` (vector, strength, components, timestamp)
  * `compressed_patterns` (profit/loss directions + strengths, compression\_count)
  * `memory_compression` (totals, utilization, efficiency, last\_compression)
  * `feature_importance` (PCA `components_`, `explained_variance_ratio`, `n_features`) — when profit PCA is fitted

---

## RAW CAPTURE — MemoryCompressor

**Class:** `MemoryCompressor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="MemoryCompressor"`, `version="3.0.0"`, `category="memory"`, `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

* `CompressorConfig`: `compress_interval=10`, `n_components=8`, `profit_threshold=10.0`, `max_memory_size=1000`, `compression_ratio=0.7`, `learning_rate=0.1`, weights: `profit_weight=2.0`, `loss_avoidance_weight=1.5`; thresholds: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_compression_quality=0.3`; feature knobs (currently unused in code paths): `feature_stability_threshold`, `pattern_confidence_threshold`.
* Genome mirrors config (same keys).

### 2) Core state & storage

* Memories: `profit_memory: List[(features, pnl)]`, `loss_memory: List[(features, |pnl|)]`.
* PCA stack: shared `StandardScaler`, `profit_pca` & `loss_pca` (`n_components`).
* Representations: `intuition_vector`, `profit_direction`, `loss_direction` (all `np.float32`).
* Tracking: `_compression_history(50)`, `_intuition_evolution(100)`, `_pattern_strength_history(200)`, `_compression_quality_scores(50)`, `_explained_variance_history(50)`.
* Perf snapshot: episodes, total compressions, avg time, memory utilization.

### 3) Data intake

* `_extract_compression_data()` reads SmartInfoBus: `trades`, `features`, `market_context`, `episode_data`; `episode` taken from inputs or episode\_data.
* Features can be `None`; `_enhance_features_with_context()` then falls back to zeros(10) and appends context (volatility/session/trend).

### 4) Memory updates

* `_process_trades_into_memory()` (last 20 trades):

  * Enhance base features with context, append per-trade `confidence` and `volume` if present.
  * Store as profit if `pnl > profit_threshold`; store as loss if `pnl < -profit_threshold/2`.
  * Tracks counts; updates utilization; `_trim_memory_buffers()` keeps most profitable/costly and recent halves when exceeding limits.

### 5) Compression cycle

* `_should_compress(episode)` → `episode % compress_interval == 0 and episode > 0`.
* `_perform_compression()`:

  * **Profit path:** weight by normalized profits → fit `StandardScaler` (first time), standardize → fit `profit_pca` → transform → weighted avg → `profit_direction`; record `explained_variance`.
  * **Loss path:** weight losses → **reuse same scaler** to transform → fit `loss_pca` → transform → weighted avg → `loss_direction`; record `explained_variance`.
  * Record event, increment `_compression_count` and totals.

### 6) Intuition update

* `_update_intuition_vector()`: blend `+ profit_weight * profit_direction` and `- loss_avoidance_weight * loss_direction`, EMA with `learning_rate`, then L2 normalize; track evolution (strengths + timestamp).

### 7) Output / returns

* `process()` returns detailed `memory_result` plus a **provides-contract** bundle:

  * `intuition_vector` (vector, strength, components, last\_updated)
  * `compressed_patterns` (profit/loss directions + strengths, compression\_count)
  * `memory_compression` (totals, utilization, compression\_efficiency, last\_compression)
  * `feature_importance` (from **profit** PCA only)
  * `_thesis` string from `_generate_compression_thesis()` (mem counts, utilization, EVR, quality, learning signals)

### 8) Health, errors, lifecycle

* Background monitor every 30s: `_update_compression_health()` sets status `warning` if avg quality < `min_compression_quality` or utilization > 95%; `_analyze_compression_efficiency()` logs improving trend.
* Error path: increments CB, opens after 3, logs with `ErrorPinpointer` + English explanation; returns fallback with last known state and provides bundle; **resets CB on success**.
* State persistence: `get_state()` (last 100 memories) / `set_state()`; `stop_monitoring()` to end thread.
* Legacy API:

  * `propose_action()` returns first 1–2 dims of intuition as action + confidence = ‖intuition‖.
  * `calculate_confidence()` combines intuition strength (0.4) + quality (0.3) + utilization (0.2) + CB factor (0.1).
  * `confidence()` (legacy) = ‖intuition‖.

---

## Quick integration notes / gotchas

* **Compression cadence:** nothing happens until `episode > 0` and is a multiple of `compress_interval`.
* **Feature dimensionality:** since the **same StandardScaler** is used for both profit & loss streams (fitted on profit first), keep feature dimensionality stable across runs or re-fit carefully to avoid transform errors.
* **Background threads:** each instance spawns its own monitor; call `stop_monitoring()` on teardown.
* **Efficiency metric:** `_compression_efficiency` is tracked but never updated in code paths (will report `0.0` unless you add logic).
* **Unused config:** `feature_stability_threshold`, `pattern_confidence_threshold` are defined but not used.
* **Requires vs tolerance:** Declared requires are read; module degrades gracefully when data is missing (falls back to cached intuition).

---

want me to capture another file the same way?
got it — here’s the clean capture + ledger entry for **MistakeMemory** in the same format.

---

## 📒 Master ledger (append)

**Module:** MistakeMemory
**File:** `modules/memory/mistake_memory.py`
**Category:** memory
**Version:** 3.0.2
**Provides:**
`mistake_memory`, `mistake_avoidance`, `danger_zones`, `pattern_recognition`, `loss_prevention`
**Requires (declared):**
`trades`, `features`, `market_context`
**Optionally reads (not required):**
`time_risk_analysis` (preferred) or legacy `risk_data`
**Explainable:** yes (`thesis_required=True`; `_thesis` returned & used in bus writes)
**Priority / timing hints:** none declared (no `timeout_ms` / `priority`)
**Concurrency / background:** per-instance monitoring thread (`_start_monitoring`, 30s loop)
**Perf tracking:** `performance_tracker.record_metric("MistakeMemory","learning_cycle", …)` on success/failure
**Circuit breaker:** threshold `3` (OPEN on repeated errors; auto-resets on success)

**Side-effects (SmartInfoBus writes):**

* On init:

  * `mistake_avoidance` (initial danger/profit zones, avoidance signal, counters)
  * `mistake_memory` (compact summary: `current_score`, `consecutive_losses`, `avoidance_signal`, `last_updated`)
* Each `process()` (comprehensive update):

  * `mistake_avoidance` (signal, counts, totals)
  * `danger_zones` (centers, count, sensitivity, timestamp)
  * `pattern_recognition` (top 10 loss/win patterns + totals)
  * `loss_prevention` (effectiveness, FPR/TPR, cluster quality, sample count)
  * `mistake_memory` (compact summary for consumers)

---

## RAW CAPTURE — MistakeMemory

**Class:** `MistakeMemory(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="MistakeMemory"`, `version="3.0.2"`, `category="memory"`, `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

* `MistakeConfig` defaults:
  `max_mistakes=100`, `n_clusters=5`, `profit_threshold=10.0`, `cluster_update_threshold=10`, `avoidance_sensitivity=1.0`, `pattern_memory_size=50`, `danger_zone_weight=2.0`; perf/circuit: `max_processing_time_ms=250`, `circuit_breaker_threshold=3`, `min_cluster_quality=0.3`; learning knobs: `learning_rate=0.1`, `false_positive_threshold=0.2`, `min_samples_for_clustering=10`.
* Genome mirrors config (same keys).

### 2) Core state & storage

* Buffers: `_loss_buf: List[(features, |loss|, trade_info)]`, `_win_buf: List[(features, profit, trade_info)]`.
* Models: shared `_scaler = StandardScaler()`, `_km_loss` & `_km_win` (KMeans).
* Zones: `_danger_zones`, `_profit_zones` (cluster centers in **scaled space**).
* Tracking/metrics: `_consecutive_losses`, `_loss_patterns`, `_win_patterns`, `_avoidance_signal`, `_cluster_quality_scores(20)`, `_learning_effectiveness(200)`, performance dict (totals, cluster updates, pattern discoveries).

### 3) Data intake & learning

* `_extract_learning_data()` pulls `trades`, `features`, `market_context`, plus risk intel (`time_risk_analysis` preferred, falls back to `risk_data`).
* `_process_learning_data()` ingests last 20 trades; builds feature vectors from trade + context; stores **significant** losses (`pnl < -profit_threshold/2`) and wins (`pnl > profit_threshold`) into respective buffers; updates counters & patterns via `_extract_pattern()` (simple discretization + action tag).

### 4) Clustering cadence & quality

* `_update_clustering()` triggers when `(len(loss)+len(win)) % cluster_update_threshold == 0` and > 0.
* Loss clustering (`_cluster_loss_data`): fit `_scaler` on loss features; KMeans with `min(n_clusters, n_samples)`; silhouette score if >1 label; `danger_zones = cluster_centers_` (scaled space); severity per cluster via mean loss.
* Win clustering (`_cluster_win_data`): **reuses** the same `_scaler` via `.transform(...)`; KMeans; silhouette; `profit_zones = cluster_centers_`; profitability per cluster.

### 5) Avoidance signal & outputs

* `_calculate_avoidance_signals()` computes similarity of current features to zones (inverse distance in scaled space), then:
  `avoidance_signal = danger_similarity * avoidance_sensitivity` (halved if profit\_similarity > danger\_similarity), scaled up for streaks of consecutive losses.
* `process()` returns rich `learning_result` plus **provides-contract** bundle:

  * `mistake_memory` (compact summary with `[0,1]` bounded `current_score` = `abs(avoidance_signal)`),
  * `mistake_avoidance`, `danger_zones`, `pattern_recognition`, `loss_prevention`, and `_thesis`.

### 6) Health, errors, lifecycle

* Background monitor (30s): `_update_mistake_health()` (warning if avg cluster quality < `min_cluster_quality` or streak > 5) and `_analyze_avoidance_effectiveness()` (logs improvement).
* Error path: circuit breaker accounting + English explanation; returns fallback with last known state and full provides bundle; **resets CB on success**.
* State persistence: `get_state()` / `set_state()`; `stop_monitoring()` to end thread.

### 7) Legacy API

* `check_similarity_to_mistakes(features)` → danger similarity.
* `propose_action()` returns avoidance action `[-avoidance_signal, 0.0]`; confidence ≈ `1 - avoidance_signal`.
* `calculate_confidence()` blends base (`1 - avoidance_signal`), cluster quality, memory size, CB factor, minus penalty for consecutive losses.
* `confidence()` returns `max(0, 1 - avoidance_signal)`.

---

## Quick integration notes / gotchas

* **Scaler fit order:** `_scaler` is **fitted in loss clustering** and then reused for win clustering. If you have enough **win** samples to trigger clustering but **insufficient loss** samples, `_scaler.transform()` in `_cluster_win_data()` may be called **before** any fit — which would raise an error. Easiest guard: ensure loss clustering runs at least once before win clustering, or add a fit-on-win fallback.
* **Zone coordinates:** `danger_zones` / `profit_zones` are **in scaled feature space** (StandardScaler). Keep this in mind if you compare against raw features elsewhere.
* **Clustering cadence:** updates only when `total_samples % cluster_update_threshold == 0`; if you trickle data in, clustering won’t update until the threshold boundary.
* **Unused/underused config fields:** `danger_zone_weight`, `false_positive_threshold` are defined but not actively used in calculations yet.
* **Threading:** each instance spawns its own monitor; call `stop_monitoring()` on teardown to avoid orphan threads.
* **Feature length:** `_extract_trade_features` pads to at least 8 and caps at 20 features. If upstream feature schema changes, clustering dimensionality must stay consistent across runs.

---

want me to capture another file the same way?
got it — here’s the clean capture + ledger entry for **NeuralMemoryArchitect** in the same format.

---

## 📒 Master ledger (append)

**Module:** NeuralMemoryArchitect
**File:** `modules/memory/neural_memory_architect.py`
**Category:** memory
**Version:** 3.0.0
**Provides:** `neural_memory`, `attention_retrieval`, `memory_embedding`, `importance_scoring`
**Requires (declared):** `observations`, `rewards`, `actions`, `market_context`
**Explainable:** yes (`thesis_required=True`; `_thesis` included in returns & bus writes)
**Concurrency / background:** per-instance monitor thread (`_start_monitoring`, 30s loop)
**Perf tracking:** `performance_tracker.record_metric("NeuralMemoryArchitect","neural_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN after repeated errors; resets on success)
**Device:** CPU (PyTorch)

**SmartInfoBus writes (side-effects):**

* On init: `neural_memory` (buffer size, utilization, avg importance, perf score).
* Each `process()`:

  * `neural_memory` (status & perf score)
  * `attention_retrieval` (only if retrieval ran: count, similarity scores, k, heads)
  * `memory_embedding` (dim, totals, thresholds/decay)
  * `importance_scoring` (stats; only when scores exist)

---

## RAW CAPTURE — NeuralMemoryArchitect

**Class:** `NeuralMemoryArchitect(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="NeuralMemoryArchitect"`, `category="memory"`, `version="3.0.0"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

`NeuralMemoryConfig` defaults:

* Core: `embed_dim=32`, `num_heads=4`, `max_len=500`, `memory_decay=0.95`, `importance_threshold=0.3`, `retrieval_top_k=5`, `learning_rate=0.001`, `attention_dropout=0.1`
* Perf: `max_processing_time_ms=400`, `circuit_breaker_threshold=3`, `min_memory_quality=0.4`
* NN extras: `hidden_multiplier=2`, `context_features=8`, `quality_threshold=0.5`
  Genome mirrors these keys.

### 2) State & metrics

* Storage: `buffer: [N, embed_dim] (torch)`, `importance_scores: [N] (torch)`, `memory_metadata: List[dict]`.
* Histories: `_memory_usage_history (200)`, `_retrieval_history (100)`, `_importance_evolution (500)`, `_attention_patterns (50)`.
* KPIs: `_storage_efficiency`, `_retrieval_accuracy`, `_memory_turnover_rate`, `_neural_performance_score` (0–100).
* Perf counters: `memories_stored`, `memories_retrieved`, `average_importance`, `attention_efficiency`.

### 3) Networks

* `encoder`: Linear → ReLU → Dropout(0.1) → Linear → LayerNorm (maps `embed_dim`→`embed_dim*hidden_multiplier`→`embed_dim`).
* `attn`: `nn.MultiheadAttention(embed_dim, num_heads, dropout=attention_dropout, batch_first=True)`.
* `value_head`: MLP + Sigmoid → scalar importance ∈ \[0,1].
* `context_net`: fuses `embed_dim + context_features` → `embed_dim` (currently not used during retrieval).
* Weights initialized with Xavier; device fixed to CPU.

### 4) Data path

* `_extract_memory_data()` reads bus: `observations`, `rewards`, `actions`, `market_context` (+ optional `experience`/`query` from inputs).
* Storage (`_process_experience_storage`):

  * Feature build from `experience` (+ `_extract_context_features`); pad/clip to `embed_dim`.
  * Encode via `encoder` (no grad).
  * Score via `value_head` (no grad); reward-conditioned boost/penalty (recent 5 rewards).
  * Store only if `importance > importance_threshold`.
  * Tracks running `average_importance` and `memories_stored`.
  * Pruning (`_prune_buffer`): keep \~80% of `max_len` using top-importance + most recent.
* Retrieval (`_perform_memory_retrieval`):

  * Pads/truncates `query` to `embed_dim`.
  * Attention over entire `buffer`; top-k by attention weights; returns embeddings + importance + metadata.

### 5) Metrics, thesis, health

* `_update_neural_metrics()`: storage efficiency, utilization, perf score (0–100).
* `_generate_neural_thesis()`: human summary (buffer, utilization, thresholds, retrieval count, heads, high-importance count, trend).
* Health monitor: warns if perf score < 50 or utilization > 95%; circuit breaker on errors; full fallback payloads.

### 6) Legacy APIs

* `retrieve(query, top_k=None)` runs async retrieval in a temporary event loop.
* `propose_action()`: uses most-important memory’s first 1–2 dims as action; confidence = perf\_score/100.
* `calculate_confidence()`: blends perf score, utilization, avg importance, CB, health.

---

## Quick integration notes / gotchas

1. **Dropout active during inference (non-deterministic):**
   `encoder` and `attn` include dropout, but the module never calls `.eval()`. `torch.no_grad()` does **not** disable dropout. So encoding & retrieval can be stochastic. Fix: call `self.encoder.eval(); self.attn.eval(); self.value_head.eval(); self.context_net.eval()` after init, or set dropout to `0.0` for production.

2. **Health severity ordering bug:**
   In `_update_neural_health()`:

   ```python
   if score < 50: status='warning'
   elif score < 20: status='critical'
   ```

   The `critical` branch is unreachable. Swap checks or reorder to `if <20: critical elif <50: warning`.

3. **Unused knobs:**
   `memory_decay`, `attention_temperature`, `importance_scaling`, `decay_adjustment`, and `context_net` aren’t currently applied in storage/retrieval logic (beyond reporting). Consider wiring them in or pruning.

4. **Average importance can reflect unstored items:**
   When an experience is **not** stored, `average_importance` may still be updated (denominator stays 0 then set to the latest importance). If you want avg over stored items only, gate that update on `stored`.

5. **Pruning index edge cases:**
   `recent_indices = torch.arange(len(buffer) - n_keep//2, len(buffer))` may include negative indices when `len(buffer) < n_keep//2`. PyTorch negative advanced indices work but can be surprising; clamp start to `0` for clarity.

6. **Retrieval shapes:**
   Attention weights come back as `[1, 1, N]`; code squeezes to `[N]` (good). Just note top-k is computed on **raw attention weights** (not normalized similarity), which is fine but implies head-averaged, single-query attention.

7. **Throughput:**
   Retrieval is O(N) over the buffer (full attention). If `max_len` grows or you raise `top_k`, consider approximate search or chunked attention.

---

want me to capture another file the same way?
sweet — here’s the clean capture + ledger entry for **PlaybookMemory** in the same style.

---

## 📒 Master ledger (append)

**Module:** PlaybookMemory
**File:** `modules/memory/playbook_memory.py`
**Category:** memory
**Version:** 3.0.1
**Provides:** `playbook_recall`, `pattern_memory`, `playbook_quality`, `memory_analytics`
**Requires (declared):** `trades`, `actions`, `market_data`, `prices`
**Explainable:** yes (`thesis_required=True`; `_thesis` included in returns & bus writes)
**Concurrency / background:** per-instance monitor thread (`_start_monitoring`, 30s loop)
**Perf tracking:** `performance_tracker.record_metric("PlaybookMemory","memory_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN after repeated errors; resets on success)

**SmartInfoBus writes (side-effects):**

* On init: `playbook_recall` (entries, patterns, memory\_quality, recall\_efficiency).
* Each `process()`:

  * `playbook_recall` (entries, patterns\_identified, recall\_efficiency, prediction\_accuracy, last\_recall)
  * `pattern_memory` (pattern\_effectiveness snapshot, diversity, top\_pattern)
  * `playbook_quality` (utilization, quality\_score, models\_fitted, adaptive\_k)
  * `memory_analytics` (total\_recalls, recent\_performance, memory\_health, CB state)

---

## RAW CAPTURE — PlaybookMemory

**Class:** `PlaybookMemory(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="PlaybookMemory"`, `category="memory"`, `version="3.0.1"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

`PlaybookConfig` defaults:

* Core: `max_entries=500`, `k=5`, `profit_weight=2.0`, `context_weight=1.5`, `similarity_threshold=0.7`, `memory_decay=0.98`, `context_features_weight=0.3`, `recency_weight=0.1`
* Perf: `max_processing_time_ms=150`, `circuit_breaker_threshold=3`, `min_quality_threshold=0.5`
* Analysis: `max_recall_history=100`, `pattern_effectiveness_window=50`, `quality_prune_threshold=0.3`
  Genome mirrors the above keys.

### 2) State & metrics

* Storage: `_features: List[np.ndarray]`, `_actions`, `_pnls`, `_contexts`, `_timestamps`, `_trade_metadata`.
* Tracking: `_recall_history (max 100)`, `_pattern_effectiveness` (wins/losses/total\_pnl by context), `_context_patterns`, `_similarity_scores (windowed)`.
* KPIs: `_memory_quality_score`, `_prediction_accuracy`, `_pattern_diversity`, `_recall_efficiency`.
* ML: `_nbrs` (KNN), `_weighted_nbrs` (second KNN), `_scaler=StandardScaler`.
* Adaptive: `_adaptive_params = {dynamic_k, context_importance, profit_bias, quality_threshold}`.

### 3) Data path

* `_extract_trade_data()` reads bus: `trades`, `actions`, `market_data`, `prices`; derives a normalized `context`.
* For each trade in last batch:

  * `_extract_trade_features()` builds a \~15-dim vector:

    * regime 1-hot (3), volatility scalar (1), risk (drawdown/exposure/position\_count → 3), session 1-hot (3), trade feats (size, confidence, side → 3), price context (price, Δ% → 2).
  * `_extract_trade_action()` maps side/size → 2-D action vector.
  * `_record_trade_memory()` decays prior PnLs, appends new sample, updates pattern tracking & metrics.
* Model fit: `_fit_memory_models()` on demand (>=k samples) → fit scaler + KNN (`_nbrs`) and an extended KNN (`_weighted_nbrs`), and refresh `_memory_quality_score`.

### 4) Recall & analytics

* `_perform_memory_recall(query_features)`:

  * Ensures 2-D shape, scales with `_scaler`, queries `_nbrs` with `dynamic_k`.
  * Computes `expected_pnl = mean(neighbor pnls)` and `confidence = exp(-mean(distance))`.
  * Logs to `_recall_history`.
* `_update_memory_analytics()`:

  * Utilization = size / max\_entries.
  * Diversity = min(1, unique\_patterns/20).
  * Recall efficiency = mean recent confidences.
  * “Prediction accuracy” = total wins / (wins+losses) across patterns.

### 5) Thesis & health

* `_generate_memory_thesis()` summarizes memory size/utilization, quality score, pattern stats, prediction/recall metrics, processed trades, recall outcome, and neighbor count.
* Health monitor: warns for utilization <10% or >90%, or if models should be fitted but aren’t; circuit breaker on errors; consistent fallback payloads.

### 6) Legacy APIs

* `step(...)` schedules async record (see caveat below).
* `recall(features)` runs `_perform_memory_recall` via a temporary event loop.
* `confidence(...)` wraps `calculate_confidence` similarly.

---

## Quick integration notes / gotchas

1. **Unreachable code in `calculate_confidence` (early return):**
   The function returns here:

   ```python
   return float(max(0.0, min(1.0, float(base_confidence))))
   ```

   but contains additional logic afterwards (adding `_prediction_accuracy`, `_recall_efficiency`, etc.) that never runs. Remove the first return or consolidate the logic.

2. **`asyncio.create_task` used from sync context (`step`, `set_state`):**
   Both call `asyncio.create_task(...)` without ensuring a running loop, which can raise `RuntimeError` in non-async callers. Consider:

   * wrapping with `try: loop = asyncio.get_running_loop(); loop.create_task(...) except RuntimeError: asyncio.run(...)`
   * or switching these helpers to fully sync and deferring to the module’s own async `process`.

3. **Session label mismatch (‘american’ vs ‘us’):**
   `session_encoding` uses `'american'`, while other modules/contexts often use `'us'`. If upstream provides `'us'`, it will fall back to the default vector `[0.25,0.25,0.25]`. Align keys to avoid silent mis-encoding.

4. **Many config knobs unused:**
   `similarity_threshold`, `profit_weight`, `context_weight`, `context_features_weight`, `recency_weight`, and `quality_prune_threshold` are set but not applied in recall, scoring, or pruning. Either implement or prune to reduce confusion.

5. **Second KNN (`_weighted_nbrs`) unused:**
   It’s fitted but never referenced during recall. If weighting or wider neighborhood voting was intended, wire it in (e.g., distance-weighted PnL or context-weighted blending).

6. **Potential feature-length mismatch on recall:**
   `_scaler.transform(query_features)` requires same feature dimension as training (\~15). Validate/reshape incoming `query_features` or re-extract with `_extract_trade_features` to avoid transform errors.

7. **Heuristic price scaling:**
   Feature `current_price / 2.0` is arbitrary. Consider relying on the scaler (already present) and pass raw price, or standardize price features relative to symbol stats.

8. **Model-fitness health rule:**
   Health flags `warning` if enough data exists but `_nbrs` is `None`. Good. Also consider warning when `_nbrs` exists but `_scaler` hasn’t seen data (shouldn’t happen, but defensive checks help).

9. **Imports with no use:**
   `cosine_similarity` and `random` are imported but unused.

10. **Logging paths & consistency:**
    Logger path here is `logs/playbook_memory.log` (others used `logs/memory/...`). Not wrong—just note the difference if you’re log-grepping by subsystem.

---

want me to capture another file the same way?
sweet — here’s the clean capture + ledger entry for **MetaAgent** in the same style.

---

## 📒 Master ledger (append)

**Module:** MetaAgent
**File:** `modules/meta/meta_agent.py`
**Category:** meta
**Version:** 3.0.0
**Provides:** `automation_decisions`, `system_mode`, `automation_metrics`, `meta_performance`
**Requires (declared):** `training_metrics`, `market_conditions`
**Explainable:** yes (`thesis_required=True`; `_thesis` returned and used in bus writes)
**Voting:** `is_voting_member=True`
**Concurrency / background:** monitor thread (`_start_monitoring`, 30s loop)
**Perf tracking:** `performance_tracker.record_metric("MetaAgent","automation_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN after repeated errors; resets on success)

**SmartInfoBus writes (side-effects):**

* On init: `automation_decisions` (mode, confidence, score, switches=0).
* Each `process()`:

  * `automation_decisions` (mode, confidence, score, last\_decision, mode\_duration; thesis = full narrative)
  * `system_mode` (mode, start time, transitions count, available modes)
  * `automation_metrics` (score, switches, accuracy, system\_confidence, current\_mode, mode\_duration)
  * `meta_performance` (daily\_pnl, drawdown\_pct, streaks, system\_confidence, training\_episodes)

---

## RAW CAPTURE — MetaAgent

**Class:** `MetaAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="MetaAgent"`, `category="meta"`, `version="3.0.0"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`, `is_voting_member=True`

### 1) Config & genome

`MetaAgentConfig` defaults:

* Core thresholds: `profit_target=150`, `retrain_threshold=-50`, `emergency_threshold=-100`, `confidence_threshold=0.7`
* Automation timing: `min_training_episodes=100`, `convergence_episodes=10`, `live_evaluation_period=3600`, `retrain_cooldown=7200`, `emergency_cooldown=1800`
* Perf: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_automation_score=0.4`
* Dynamics: `confidence_decay=0.98`, `performance_smoothing=0.95`, `decision_history_size=100`
  Genome mirrors config; if provided, it also mutates `self.config` to keep them in sync.

### 2) State & metrics

* Modes: `current_mode` (Enum), `mode_start_time`, `mode_transitions` (deque)
* P\&L & risk: `daily_pnl`, `session_pnl`, `peak_pnl`, `drawdown_pct`, `consecutive_losses`, `win_streak`, `loss_streak`
* Training: `training_episodes`, `best_training_reward`, `training_convergence_count`, `validation_performance` (deque)
* Automation: `system_confidence` (smoothed/decayed), `automation_score`, timestamps for last decisions/retrain/emergency
* Histories: `_performance_history`, `_confidence_history`, `_mode_performance`, `_decision_effectiveness`
* Metrics block: `automation_metrics` (switch counts, accuracy, avg durations placeholders, etc.)

### 3) Data path

* `_extract_meta_data()` pulls from bus:

  * **required:** `training_metrics`, `market_conditions` (but code tolerates missing by using `{}`)
  * **optional:** `system_performance`, `risk_signals` or `time_risk_analysis`
  * **direct inputs:** `pnl`, `performance_update`
* `_update_system_performance()`:

  * updates P\&L, drawdown, streaks
  * ingests `performance_update` (episode reward / validation score)
  * ingests `training_metrics`
  * recomputes `system_confidence` (performance/drawdown/losses/training with smoothing + decay)
* Decision flow:

  * `_evaluate_automation_decision()` → emergency guard → mode-specific transition:

    * `INITIALIZATION` → TRAINING after 60s
    * `TRAINING` → VALIDATION on (episodes ≥ min & convergences ≥ target) or after 2h timeout
    * `VALIDATION` → LIVE on avg last-10 validation > 0.7 (else no exit until len≥10)
    * `LIVE_TRADING` → EVALUATION on low confidence or profit target; → RETRAINING on loss threshold
    * `EVALUATION` → LIVE or RETRAINING after `live_evaluation_period` based on confidence
    * `EMERGENCY_STOP` → EVALUATION after `emergency_cooldown`
* `_execute_decision()` logs, records, transitions, increments counters
* `_update_automation_metrics()` computes `automation_score` with `_evaluate_decision_quality()` and fills telemetry

### 4) Returns & thesis

* `process()` returns **all four provided keys** + `_thesis` and `success` flag.
* Thesis includes: mode/duration, P\&L & confidence, decision quality score, any transition, risk alerts, automation stats, training progress, confidence trend.

---

## Quick integration notes / gotchas

1. **`HALF_OPEN` not used:**
   `calculate_confidence()` branches on CB states `OPEN`, `HALF_OPEN`, `CLOSED`, but the code never sets `HALF_OPEN`. Either implement a half-open step or drop that branch.

2. **Validation can stall:**
   In `_evaluate_validation_transition`, the timeout path (`mode_duration > 1800 → RETRAINING`) is inside the `len(self.validation_performance) >= 10` block. If you never reach 10 validation points, VALIDATION can hang indefinitely. Consider moving the timeout outside that length guard.

3. **Unused config knobs:**
   `window`, `improvement_threshold`, `retrain_cooldown` are defined but unused; `avg_training_duration` / `avg_live_duration` never computed. Either wire them up (e.g., enforce cooldowns on transitions; compute averages from `_mode_performance`) or prune.

4. **Reliance on current-day P\&L for decision quality:**
   `_evaluate_decision_quality()` scores decisions using *current* `daily_pnl`, not the *post-decision* outcome window. This can misattribute quality. Consider tagging decisions with a snapshot and grading them later against a windowed delta.

5. **“Successful live sessions” counter is lenient:**
   It increments whenever you exit LIVE\_TRADING without an emergency, regardless of profitability. If you want true success, gate on profit or positive Sharpe during the session.

6. **`np.isnan(pnl)` can crash on bad types:**
   If a caller passes a string or `None` pnl, `np.isnan` raises. Guard with `isinstance(pnl,(int,float,np.floating))` before the check.

7. **`market_conditions`/`risk_signals` are fetched but unused:**
   None of the transition logic references them. If you intend regime/volatility gating or risk overrides, plug them into the decision rules.

8. **Confidence model weights are arbitrary & unit-sensitive:**
   `performance_confidence = (avg_pnl+50)/100` implies P\&L units of \~€50 scale. If live P\&L scale differs, confidence becomes skewed. Consider normalizing by recent volatility or ATR of P\&L.

9. **Mode performance analytics exist but aren’t surfaced:**
   `_mode_performance` accumulates per-mode stats; great place to compute `avg_training_duration`, `avg_live_duration`, and feed them into `automation_metrics`.

10. **Thread/loop hygiene:**
    `step()` spins a fresh event loop per call; that’s safe but a bit heavy. If this runs in a service with an existing loop, consider detecting a running loop and scheduling `process()` onto it.

11. **Emergency logic edges:**
    Emergency triggers are based on **drawdown\_pct**, **consecutive\_losses**, **daily\_pnl**, **confidence**. If you also track *intra-day* or *session* P\&L, you might want session-level thresholds to catch fast crashes.

12. **State shape in bus writes:**
    Bus payloads are compact (counts & current mode). If downstream UI needs richer history, consider also publishing a capped recent `mode_transitions` list (timestamps converted to ISO strings).

13. **Naming consistency:**
    Logs use tags like `[RELOAD]`, `[TARGET]`, `[WARN]`. Good; just ensure log scrapers pick up the same tags used by other modules.

---

want me to capture another file the same way?
sweet — here’s the clean capture + ledger entry for **MetaCognitivePlanner** in the same style.

---

## 📒 Master ledger (append)

**Module:** MetaCognitivePlanner
**File:** `modules/meta/metacognitive_planner.py`
**Category:** meta
**Version:** 3.0.1
**Provides:** `planning_status`, `strategic_insights`, `tactical_recommendations`, `adaptation_metrics`
**Requires (declared):** `market_data`  *(others read with soft fallbacks: trades, actions, performance\_metrics, market\_regime, regime\_data, volatility\_adjustment, market\_conditions)*
**Explainable:** yes (`thesis_required=True`; `_thesis` added to return and used in bus writes)
**Voting:** no
**Concurrency / background:** idempotent monitor thread (`_start_monitoring`, 30s loop)
**Perf tracking:** `performance_tracker.record_metric("MetaCognitivePlanner","planning_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN after repeated errors; resets to CLOSED on success)

**SmartInfoBus writes (side-effects):** on each `process()`

* `planning_status` → phase, cycle, phase duration, cognitive\_load, planning\_confidence, strategy\_coherence *(thesis = full narrative)*
* `strategic_insights` → totals, recent three insights, current recommendations, planning\_effectiveness snapshot
* `tactical_recommendations` → active recommendations, session plan count, strategic\_objectives
* `adaptation_metrics` → total adaptations, adaptation\_speed, learning entries, strategy\_evolution size

---

## RAW CAPTURE — MetaCognitivePlanner

**Class:** `MetaCognitivePlanner(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="MetaCognitivePlanner"`, `category="meta"`, `version="3.0.1"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

`PlanningConfig` defaults:

* Phasing: `phase_min_duration=120s`, `phase_max_duration=600s`
* Strategy targets: `profit_target=150`, `max_drawdown=15%`, `min_win_rate=0.55`, `risk_budget=0.10`
* Planning shape: `window=20`, `planning_horizon=100`, `adaptation_threshold=0.15`
* Perf/safety: `max_processing_time_ms=200`, `circuit_breaker_threshold=3`, `min_confidence_threshold=0.6`
  Genome mirrors most knobs (`risk_budget` excluded) and seeds state.

### 2) State & metrics

* Phases: `current_phase` (Enum: ANALYSIS→PLANNING→EXECUTION→REFLECTION→ADAPTATION), `phase_start_time`, `planning_cycle`
* Buffers: `episode_history(maxlen=window)`, `session_plans(maxlen=10)`, `adaptation_history(maxlen=50)`, `strategic_insights(maxlen=20)`, `learning_history(maxlen=100)`
* Strategy data: `strategic_objectives` dict (profit target, dd cap, win-rate, risk\_budget=0.10, diversification\_target, adaptive\_learning\_rate)
* Analytics: `planning_effectiveness` (phase → wins/total/outcome), `strategy_performance`, `market_adaptation_patterns`
* Cognitive: `cognitive_load`, `planning_confidence`, `strategy_coherence`, `adaptation_speed`
* Recommendations cache: `current_recommendations`

### 3) Data path

* `_extract_planning_data()` pulls from bus (robust, canonical-aware) and builds a **normalized context** via `_extract_standard_context()` with regime/volatility/session + risk posture fields.
* Main loop in `process()`:

  1. Execute phase (`_execute_planning_phase` → dispatch)
  2. Evaluate phase transition (`_evaluate_phase_transition`)
  3. Generate insights (`_generate_strategic_insights`)
  4. Update cognitive metrics (`_update_cognitive_metrics`)
  5. Build `_thesis`, write to SmartInfoBus, and **guarantee the four provided keys** in the return

### 4) Phases (what they do)

* **ANALYSIS:** market/system assessment; stores an analysis record into `strategic_insights` (deque)
* **PLANNING:** builds a comprehensive plan (strategic plan + tactical recs + risk plan + success criteria + fallbacks) and caches `current_recommendations`
* **EXECUTION:** monitors plan (`_monitor_plan_execution`), detects deviations, produces adjustments
* **REFLECTION:** evaluates outcomes, produces lessons, updates `planning_effectiveness`, logs learning entry
* **ADAPTATION:** mines learning history for opportunities, proposes/apply adaptations (e.g., threshold shifts), records an adaptation entry

### 5) Phase transitions

* Time-boxed: hard exit after `phase_max_duration` (600s).
* Soft exit after `phase_min_duration` (120s) if phase-specific conditions met (e.g., analysis done, plan created, execution completion > 0.7, reflection logged).
* ADAPTATION always transitions after min duration (one pass); wrapping from ADAPTATION→ANALYSIS increments `planning_cycle`.

### 6) Outputs & thesis

* Returns always include:

  * `planning_status` (phase/cycle/duration + cognitive metrics)
  * `strategic_insights` (list; may be empty)
  * `tactical_recommendations` (list; fallback to `current_recommendations`)
  * `adaptation_metrics` (totals & speeds)
  * `_thesis` (phase narrative with cognitive state)
* Error/fallback paths maintain the same four keys + `_thesis` and circuit breaker state.

### 7) Actions & confidence

* `propose_action()` emits a **planning** action (e.g., `strategic_plan`, `execution_adjustment`, etc.) keyed to the current phase and confidence.
* `calculate_confidence()` blends planning\_confidence with cognitive load, coherence, and a phase bonus (EXECUTION +0.1), clipped to `[0.1, 1.0]`.

---

## Quick integration notes / gotchas

1. **`risk_budget` not in genome & not used downstream.**
   Config has `risk_budget`, but genome excludes it and plans use a hardcoded `0.10`. Consider sourcing from config or genome for consistency.

2. **`planning_horizon` currently unused.**
   It’s logged but doesn’t affect plan generation or transitions. If intended, apply to plan timelines or lookback windows.

3. **`min_confidence_threshold` unused.**
   Could gate phase transitions (e.g., block LIVE-like execution without sufficient planning\_confidence).

4. **`adaptation_threshold` only adjusted, never read.**
   Adaptation phase increments `genome["adaptation_threshold"]` but nothing consumes it. Wire it into transition criteria or risk rules.

5. **Name collision risk: `strategic_insights` (deque attr) vs return key.**
   Code is careful, but the dual use can confuse readers. Consider renaming the deque to `insight_log`.

6. **Planning phase returns `tactical_recommendations` as a count.**
   You guard this later (replace int with list), which is good. For clarity, return both: `tactical_recommendations_count` and `tactical_recommendations`.

7. **Health signal depends on `planning_effectiveness` only.**
   With no data, health stays whatever it was (default “healthy”). You might set a neutral/unknown state until evidence accrues.

8. **Monitor thread stop.**
   `stop_monitoring()` flips a flag but doesn’t join the thread; not fatal (daemon thread), just note for orderly shutdowns/tests.

9. **Execution monitoring/deviations are placeholders.**
   `_monitor_plan_execution()` and `_detect_plan_deviations()` use fixed thresholds; plug in real KPI deltas (slippage vs plan, adherence to risk rules).

10. **Context merge logic is solid but may miss pricing.**
    If tactical recs want ATR/realized vol or spreads, add those into `_extract_standard_context` and consume in `_generate_tactical_recommendations()`.

11. **Effectiveness accounting by plan phase.**
    `_update_planning_effectiveness()` indexes by `plan['phase']` (always `'planning'` in this code). If you want phase-wise stats, pass the active phase when outcomes are measured.

12. **Episode windowing.**
    `episode_history` maxlen uses `genome["window"]`. If genome changes at runtime, consider resizing the deque to match.

13. **Bus payload sizes.**
    You cap `recent_insights` to the last 3 (good). Keep the full objects small to avoid log spam (e.g., trim inside insights).

want me to capture another file the same way?
awesome — here’s the tidy capture + ledger entry for **MetaRLController** in the same style.

---

## 📒 Master ledger (append)

**Module:** MetaRLController
**File:** `modules/meta/meta_rl_controller.py`
**Category:** meta
**Version:** 3.0.0
**Provides:**
`controller_status`, `agents_performance`, `controller_training_overview`, `automation_status`, `trading_signals`, `trading_signal`, `meta_signals`, `agent_decisions`
**Requires (declared):** `market_data`
**Explainable:** yes (`thesis_required=True`; `_thesis` attached to output & used in bus writes)
**Voting:** not specified ⇒ no
**Concurrency / background:** monitor thread (daemon, 30s loop) via `_start_monitoring()`
**Perf tracking:** `performance_tracker.record_metric("MetaRLController","controller_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN after repeated errors; reset to CLOSED on success)

**SmartInfoBus writes (side-effects):** on each `process()`

* `controller_status` → mode, active agent, session PnL, episode, mode duration, last transition *(thesis = narrative)*
* `agents_performance` → performance summary (per agent), `best_agent`, and last `agent_comparison`
* `controller_training_overview` → last 10 training history entries, last 5 validation results, convergence/poor-episode counts
* `automation_status` → automation\_metrics snapshot, recent mode transitions and decision history

---

## RAW CAPTURE — MetaRLController

**Class:** `MetaRLController(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="MetaRLController"`, `category="meta"`, `version="3.0.0"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

`ControllerConfig` defaults:

* Core: `obs_size=64`, `act_size=2`, `method="ppo-lag"`, `device="cpu"`
* Targets & counts: `profit_target=150`, `training_episodes=1000`, `validation_episodes=100`
* Perf/safety: `max_processing_time_ms=250`, `circuit_breaker_threshold=3`, `min_convergence_threshold=0.8`
* Automation: `min_training_episodes=50`, `validation_success_rate=0.6`, `live_trading_confidence=0.7`,
  `retraining_trigger_loss=-50`, `emergency_stop_loss=-100`, `optimization_interval=500`
  Genome mirrors main knobs (`obs_size`, `act_size`, `method`, `profit_target`, episode counts).

### 2) State & agents

* Modes: `ControllerMode` (INITIALIZATION → TRAINING → VALIDATION → LIVE\_TRADING → RETRAINING → OPTIMIZATION → EMERGENCY\_STOP).
* Tracking: `mode_start_time`, `mode_transitions(maxlen=100)`, `decision_history(maxlen=200)`.
* Agents: attempts to init `PPOAgent` and `PPOLagAgent`; sets `active_agent` to preferred or fallback.
* Performance: `AgentPerformanceTracker(window=100)` for rewards/losses, convergence & stability scores; `training_history(maxlen=1000)`, `validation_results(maxlen=50)`.
* Live stats: `live_session_pnl`, `live_session_trades`, `live_performance_history(maxlen=100)`.

### 3) Data path

* `_extract_controller_data()` pulls trades/actions/market/training + builds normalized context.
* `_update_performance_metrics()` aggregates PnL/trades; updates training metrics (when in TRAINING/RETRAINING); feeds mixin `_update_trading_metrics`.
* **Main `process()`** (as written) **does not** call the above orchestration; it builds summaries from current state, computes a default `trading_signal` (`hold`), updates the bus, and returns the eight provided keys (plus `_thesis`).

### 4) Modes (execution helpers)

Helpers exist for each mode (`_execute_*_mode`) and for transition logic (`_evaluate_*_transition`), including optimization (agent comparison + switch), retraining, and emergency stop. Transitions are recorded with reason/priority and reset per-mode state.

### 5) Outputs & thesis

`process()` returns exactly the declared keys:

* `controller_status`, `agents_performance`, `controller_training_overview`, `automation_status`, `trading_signals` (list), `trading_signal` (single), `meta_signals` (dict), `agent_decisions` (list), `_thesis`.
  Thesis string defaults to `"MetaRLController status update"`; there’s also `_generate_controller_thesis()` available but not invoked in `process()`.

### 6) Actions & confidence

* `act(obs_tensor)` routes to active agent, clamps or zeros by mode; safe NaN handling.
* `record_step(...)` and `end_episode(...)` proxy to active agent and tracker.
* `propose_action()` emits controller-level intentions (continue/monitor/agent\_switch/safety\_hold).
* `calculate_confidence()` blends mode, live PnL, and agent convergence/stability.

---

## Quick integration notes / gotchas

1. **Main loop bypass.**
   `process()` currently skips the controller’s core logic (`_extract_controller_data`, `_execute_current_mode`, `_evaluate_mode_transition`, `_update_performance_metrics`, `_update_automation_metrics`, and `_generate_controller_thesis`). If this module is meant to *drive* modes, wire those calls into `process()` (or ensure an external orchestrator calls them every tick).

2. **Key name mismatches in fallbacks.**
   `_handle_no_data_fallback()` and `_create_fallback_response()` return `agent_performance` and `training_metrics`, but the declared/provided keys are `agents_performance` and `controller_training_overview`. Standardize those to avoid downstream contract breaks.

3. **Avg training duration bug.**
   `_update_automation_metrics()` computes `avg_training_duration` via `(timestamp - timestamp)` which is always `0`. You likely intended `(end_ts - start_ts)` or to accumulate durations per session.

4. **Monitoring before `super().__init__()`.**
   `_start_monitoring()` is called prior to the base class init; it’s safe here but can introduce race conditions in other bases. Consider moving after `super().__init__()` (like your other modules) or gate on a fully-initialized flag.

5. **Event loop requirement in `force_mode_transition()`.**
   Uses `asyncio.create_task(...)` from a sync method; this requires a running loop in the caller’s thread. If this can be called from non-async contexts, use `asyncio.get_running_loop()` guard or a thread-safe submission.

6. **`device` and agent wiring.**
   `ControllerConfig.device` isn’t used, and agents are constructed without config. If you need device/obs/act sizes to flow into agents, pass them at init.

7. **Agent None-safety after switch.**
   `_switch_active_agent()` doesn’t guard against `self._agents[new_agent_name] is None`. Later calls assume a callable agent. Add a check and abort switch (with log) if the target agent failed to initialize.

8. **Optimization selection heuristics.**
   `_select_optimal_agent()` mixes context preferences and tracker stats; good. Ensure performance tracker has enough warmup (`episodes > 10`) before switching to avoid churn.

9. **Health & automation signals.**
   Health toggles on live PnL thresholds and emergency mode; consider factoring in validation success rate and convergence to avoid “always healthy” during TRAINING when data is stale.

10. **Observation dimension contract.**
    `obs_dim()` returns `obs_size + 4` (controller extras). Ensure the active agent’s `get_observation_components()` aligns with `obs_size`, or normalize/clip when concatenating.

11. **Thread lifecycle.**
    `stop_monitoring()` only flips the flag; fine for daemons, but tests may want a join to confirm termination.

12. **Thesis quality.**
    There’s a rich `_generate_controller_thesis(...)`; consider calling it inside `process()` to emit a more informative `_thesis` instead of the static string.

want me to capture another file the same way?
boom — here’s the tidy capture + ledger entry for **PPOAgent** in the same style.

---

## 📒 Master ledger (append)

**Module:** PPOAgent
**File:** `modules/meta/ppo_agent.py`
**Category:** meta
**Version:** 3.0.0
**Provides:**
`policy_actions`, `agent_performance`, `training_metrics`, `policy_gradients`, `actions`, `training_data`, `observations`, `rewards`, `training_signals`
**Requires (declared):** `market_data`
**Explainable:** yes (`thesis_required=True`; `_thesis` always included)
**Voting:** yes (`is_voting_member=True`)
**Concurrency / background:** monitor thread (daemon, 30s loop) via `_start_monitoring()`
**Perf tracking:** `performance_tracker.record_metric("PPOAgent","processing_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN on repeated errors; reset to CLOSED on success)

**SmartInfoBus writes (side-effects):** per-step via `_update_ppo_smart_bus()`

* `policy_actions` → action, log\_prob, value\_estimate, action\_std, exploration\_level *(thesis = narrative)*
* `actions` → raw action vector (compat)
* `agent_performance` → perf score, avg reward, episodes, updates, learning rate
* `training_metrics` → (when training happened) policy/value/entropy losses, grad\_norm, explained\_variance, total\_updates
* `policy_gradients` → grad\_norm, learning\_rate, network\_parameters, forward/backward pass counts
* `observations` → last observation snapshot (only if stored as `np.ndarray`)
* `rewards` → recent rewards window (if any)
* `training_signals` → grad\_norm, explained\_variance, policy\_loss, value\_loss
* `training_data` → buffer sizes, total\_updates

---

## RAW CAPTURE — PPOAgent

**Class:** `PPOAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="PPOAgent"`, `category="meta"`, `version="3.0.0"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`, `is_voting_member=True`

### 1) Config & genome

`PPOConfig` defaults:

* Core: `obs_size=10`, `act_size=2`, `hidden_size=64`, `learning_rate=3e-4`, `device="cpu"`
* PPO: `clip_eps=0.2`, `value_coeff=0.5`, `entropy_coeff=0.01`, `gae_lambda=0.95`, `gamma=0.99`, `max_grad_norm=0.5`, `ppo_epochs=4`, `batch_size=64`
* Perf/safety: `max_processing_time_ms=500`, `circuit_breaker_threshold=3`, `min_performance_score=0.3`
* Training: `buffer_size=2048`, `early_stopping_patience=100`, `lr_decay_patience=50`
  Genome can override and also *pushes* values back into `self.config`.

### 2) Networks & optimizer

`EnhancedPPONetwork`

* Shared MLP (+Dropout) ➜ actor head outputs `[mean, log_std]` (size `act_size * 2`), critic head outputs value.
* Orthogonal init; policy output layer gain=1; clamps `log_std` to \[-20, 2].
  Optimizer: `Adam(lr=config.learning_rate, eps=1e-5, weight_decay=1e-4)`
  Scheduler: `ReduceLROnPlateau(mode='max', factor=0.8, patience=lr_decay_patience)` driven by **avg reward**.

### 3) State & buffers

* Experience buffer dict: `observations/actions/log_probs/values/rewards/advantages/returns/dones`.
* Rolling deques for episode rewards/lengths and for policy/value/entropy losses.
* `training_stats`: updates, episodes, best/avg reward, trends, grad\_norm, explained\_variance, current LR.
* Action tracking: `last_action`, `action_history(maxlen=1000)`, `action_statistics` (mean/std/range/exploration\_level).
* Market/context hooks: `market_context_history(maxlen=50)`, `context_performance`.
* Monitoring metrics: `_neural_performance` counters.
* Snapshots for bus: `_last_obs_vec`, `_last_action_std`, `_recent_rewards`.

### 4) Process loop

`process(**inputs)`:

* `_extract_ppo_data()` builds a data bundle **without** pulling agent’s own outputs from the bus (avoids self-dependency). Accepts direct `observation` and/or `experience` from inputs, plus `market_data` from the bus.
* If `observation` key exists (even `None`), runs `_process_action_selection()` (zeros if `None`) to produce action/log\_prob/value/std and updates local snapshots.
* If `experience` present, appends to buffer and, when `len(observations) >= batch_size`, calls `_perform_policy_update()`:

  * GAE via `_compute_gae_returns()`
  * Multi-epoch PPO update with clipped objective, value loss, entropy bonus, grad clip, tracks trends & `explained_variance`
  * Clears buffer afterward
* `_update_agent_metrics()` computes perf score (normalized avg recent reward), steps LR scheduler, updates stats.
* Ensures **all 9 provided keys** exist in the returned dict and appends `_thesis` from `_generate_ppo_thesis(...)`.
* Calls `_update_ppo_smart_bus(...)` to publish.

### 5) Legacy & helpers

* `select_action(obs_tensor)` (legacy): spins a **new event loop**, awaits `_process_action_selection`, returns tensor; caches `_last_log_prob`/`_last_value`.
* `record_step(...)`/`end_episode(...)` push into buffers & episode metrics (episode sum reward).
* `propose_action(...)`: deterministic actor-mean action (no sampling) with quick stats.
* `confidence(...)` & `calculate_confidence(...)`: blend performance, action magnitude/variance, stability (loss std), exploration.
* Health/monitor thread updates learning signals and logs notable trends.

---

## Quick integration notes / gotchas

1. **Inference mode & dropout.**
   `_process_action_selection()` uses `torch.no_grad()` but never calls `self.network.eval()`. With Dropout layers, actions will remain stochastic at inference. Consider `self.network.eval()` for action selection and `self.network.train()` only during updates.

2. **Action selection always triggers.**
   Because `process()` passes the `'observation'` key unconditionally, `_process_action_selection()` runs even when no fresh obs was provided (it’ll use a zero vector). If you only want to act on real inputs, guard with `if ppo_data.get('observation') is not None:`.

3. **Event loop creation in `select_action()`.**
   Spinning a new event loop per call is expensive and can clash with existing loops. Prefer a purely sync path for legacy (`with torch.no_grad()` forward) or reuse the running loop.

4. **GAE dependencies.**
   `_compute_gae_returns()` requires `values` and `dones` lengths to match `rewards`. Ensure callers populate `value` and `done` in `experience` (or via the legacy path setting `_last_value`) to avoid misaligned arrays.

5. **Episode buffer handling.**
   `end_episode()` sums rewards but doesn’t clear the current episode’s entries from the rolling buffer; PPO updates do clear **after** an update. If you call `end_episode()` without updating, those rewards remain accumulated in the buffer.

6. **Bus snapshots typing.**
   `observations` are only published to the bus when `_last_obs_vec` is an `np.ndarray` (lists don’t get published). If a downstream consumer relies on `observations`, ensure `_last_obs_vec` is stored as `np.ndarray` (the legacy `select_action` does this; other paths might not).

7. **Learning-rate scheduler signal.**
   Scheduler is stepped with **avg reward** (maximize). That’s fine, but if your reward scale changes across tasks, you may get premature LR reductions.

8. **Zero-observation actions.**
   If no obs are flowing, you’ll still sample actions from the network given a zero vector (which may be fine for exploration but can produce noise on startup). Gate this if undesired.

9. **Device/config flow.**
   `device` is read from config; agents created by other modules should pass the correct device in `config` (no implicit propagation from controller).

10. **Exploration metric.**
    `exploration_level` derives from the **std of recent actions**, not from the policy distribution’s entropy. That’s a decent proxy but can diverge from actual policy entropy under heavy clipping.

11. **Thread lifecycle.**
    `stop_monitoring()` just flips a flag (daemon thread ends naturally). Tests may want a join for deterministic shutdown.

12. **Bus key surface area.**
    This agent writes common keys like `actions`, `observations`, `rewards`. Your other modules (e.g., controller) already avoid reading those from this agent to prevent feedback loops—keep that convention.

want me to capture the `PPOLagAgent` next?
boom — tidy capture + ledger entry for **PPOLagAgent** coming right up.

---

## 📒 Master ledger (append)

**Module:** PPOLagAgent
**File:** `modules/meta/ppo_lag_agent.py`
**Category:** meta
**Version:** 3.0.0
**Provides:** `agent_status`, `ppo_lag_training_metrics`, `position_metrics`, `market_adaptation`
**Requires (declared):** `trades`, `actions`, `market_data`, `training_signals`
**Explainable:** yes (`thesis_required=True`; `_thesis` included in returns)
**Voting:** no (not marked as voting member)
**Concurrency / background:** monitor thread (daemon, 30s loop) via `_start_monitoring()`
**Perf tracking:** `performance_tracker.record_metric("PPOLagAgent","agent_cycle", …)`
**Circuit breaker:** threshold `3` (OPEN on repeated errors; reset to CLOSED on success)

**SmartInfoBus writes (side-effects):** via `_update_agent_smart_bus()`

* `agent_status` → kind, episodes, total\_updates, network param count, training\_active, buffer\_size *(thesis = narrative)*
* `ppo_lag_training_metrics` → training\_stats (actor/critic loss trends, KL, entropy, explained\_variance…), last episode rewards/lengths, adaptive params (clip\_eps, lr\_factor, entropy\_coeff)
* `position_metrics` → current\_position, unrealized\_pnl, risk\_metrics (max/avg/volatility/risk\_adjusted\_return), history size
* `market_adaptation` → lag\_window, lag buffer sizes, per-regime/volatility performance snapshots, adaptation settings (vol\_scaling, position\_aware)

---

## RAW CAPTURE — PPOLagAgent

**Class:** `PPOLagAgent(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator:** `name="PPOLagAgent"`, `category="meta"`, `version="3.0.0"`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

### 1) Config & genome

`PPOLagConfig` defaults:

* Core: `obs_size=64`, `act_size=2`, `hidden_size=128`, `lr=1e-4`, `lag_window=20`, `device="cpu"`
* Extras: `adv_decay=0.95`, `vol_scaling=True`, `position_aware=True`
* PPO-ish: `clip_eps=0.1`, `value_coeff=0.5`, `entropy_coeff=0.001`, `gae_lambda=0.95`, `gamma=0.99`, `max_grad_norm=0.5`, `ppo_epochs=4`, `batch_size=64`, `target_kl=0.01`
* Perf/safety: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_episode_length=10`
  Genome overrides are supported and stored in `self.genome`.

### 2) Network & optimizers

`MarketAwarePPONetwork(obs_size, act_size, lag_window, hidden_size=128)`:

* Extended obs = `obs_size + lag_window*4 + 6` (lag features: returns, volatility, volume, spread; +6 position features if enabled)
* `BatchNorm1d` on extended obs (skipped for batch size 1)
* Market encoder (linear → LN → ReLU → Dropout)
* Feature extractor (linear → LN → ReLU → Dropout → linear → ReLU)
* Actor path: Multi-head attention (4 heads) over features → concat with market features → actor head (Tanh) outputs **means** (bounded)
* Std: base learnable `log_std_base` + market-conditioned adjustment via `std_conditioner`; final `action_std` clamped `[0.01, 1.0]`
* Dual critics `value_head_1/2`, conservative value = `min(v1, v2)`
* Orthogonal init; LN/BN weights to 1, biases 0
  Optimizers:
* **Actor:** Adam over extractor + attention + actor head + std params (lr=`lr`)
* **Critic:** Adam over market encoder + both value heads (lr=`2*lr`)

### 3) State, buffers & tracking

* Lag buffers (deques): `price`, `volume`, `spread`, `volatility` (len `lag_window`)
* Experience buffer dict: `observations`, `market_features`, `actions`, `log_probs`, `values`, `rewards`, `advantages`, `returns`, `dones`
* Episode tracking: `episode_rewards/lengths` (len 200)
* Per-regime / per-volatility performance dicts
* Position & risk: `position`, `unrealized_pnl`, `position_history`, `risk_metrics` (max/avg position, position\_volatility, risk\_adjusted\_return)
* Training stats: updates, episodes, actor/critic loss trends, KL, entropy, explained\_variance, advantage stats
* Adaptive params: `running_adv_std`, `adaptive_clip_eps`, `adaptive_lr_factor`

### 4) Main loop

`process(**inputs)`:

1. `_extract_agent_data()` pulls `trades`, `actions`, `market_data`, `training_signals` from bus + direct inputs (`obs_vec`, `reward`, `done`) and builds a context (regime/vol/session/price/volume/spread/vol).
2. `_update_market_buffers(agent_data)` → validates and updates lag buffers; reports sizes.
3. `_process_training_step(agent_data)` → if `obs_vec` & `reward` present, builds extended obs (obs + lag features + position features), forward pass for `action_mean/std/value` (with vol scaling & position penalty on std), samples action, appends tensors to buffer, updates position/risk, trading metrics. If `done`, calls `_end_training_episode()`.
4. `_adapt_to_market_conditions(agent_data)` → adjusts `adaptive_clip_eps`, per-optimizer LR (via `adaptive_lr_factor` by regime), and `entropy_coeff` by volatility level.
5. `_update_performance_tracking(agent_data)` → rolls episode-level reward summaries by regime/volatility.
6. Generates thesis and publishes to bus; returns a payload that **includes all provided keys**.

### 5) Training pipeline (episode-end)

* `_compute_gae_advantages()` → GAE with bootstrap from critic, resets at dones, adaptive running-std normalization (controlled by `adv_decay`).
* `_perform_ppo_updates()` → multi-epoch, shuffles, mini-batches; Normal(action\_mean, action\_std); ratio + clipping with **adaptive** `clip_eps`; entropy bonus; early stop on large mean KL surrogate `(old_log_prob - new_log_prob).mean()`; actor & critic updated with separate optimizers; grad clip per head; aggregates losses/entropy/KL; returns averages.
* `_update_training_statistics(update_stats)` → EMA trends, explained\_variance using numpy variance on returns vs values.

### 6) Risk & observation helpers

* `_update_risk_metrics()` computes max/mean/volatility of recent positions and risk-adjusted return.
* `get_observation_components()` returns a compact 14-element vector of normalized position/risk/training/adaptation/buffer state signals.

### 7) Legacy methods

* `record_step(obs_vec, reward, **market_data)` → wraps into agent\_data and schedules `_record_training_step` with `asyncio.create_task`.
* `end_episode()` → schedules `_end_training_episode`.
* `select_action(obs_tensor)` → pads/truncates to extended\_obs\_size, extracts trailing lag slice, forward pass to sample/clamp action in `[-2, 2]`.
* `propose_action(obs_vec?)` → draws Normal(mean, std), clamps to `[-1,1]`, confidence from std, value estimate.

### 8) Health & errors

* Health: recent episode reward banding; detects NaNs in network params; updates `last_check`.
* Circuit breaker: increments on errors in process; OPEN at threshold.
* Fallbacks: `_handle_no_data_fallback()` and `_create_fallback_response()` return minimal status + circuit breaker state.
* Thesis: `_generate_agent_thesis(...)` summarizes episodes/updates, avg reward, market adaptation, position/risk snapshot.

---

## ⚠️ Integration notes / gotchas (actionable)

1. **Price buffer logic is wrong (returns vs prices).**
   `update_market_buffers()` computes `price_return` using `last_price = self.price_buffer[-1]`, but `price_buffer` stores **returns**, not prices. That makes the next return `(price - last_return)/last_return`, which is invalid.
   **Fix (minimal):** keep a `self._last_price` (float) and compute return from that; store the *return* (or decide to store prices and rename).

   ```python
   # in _initialize_agent_state
   self._last_price = None

   # in update_market_buffers
   if self._last_price is None or self._last_price <= 0:
       price_return = 0.0
   else:
       price_return = (price - self._last_price) / self._last_price
   self._last_price = price
   self.price_buffer.append(price_return)
   ```

2. **NaN sanitization in `_perform_ppo_updates()` doesn’t stick.**
   The loop

   ```python
   for name, tensor in [('observations', observations), ...]:
       if torch.any(torch.isnan(tensor)):
           tensor = torch.nan_to_num(tensor)
   ```

   assigns to a local `tensor` only. The originals remain unchanged.
   **Fix:** reassign each variable explicitly:

   ```python
   observations = torch.nan_to_num(observations)
   market_features = torch.nan_to_num(market_features)
   actions = torch.nan_to_num(actions)
   old_log_probs = torch.nan_to_num(old_log_probs)
   returns = torch.nan_to_num(returns)
   advantages = torch.nan_to_num(advantages)
   ```

3. **Entropy coeff mutated in place.**
   `_adapt_to_market_conditions()` updates `self.config.entropy_coeff` on the fly. That’s fine, but be aware this persists across regimes and threads; consider storing a base value and deriving a working value per step to avoid drift.

4. **KL check uses mean log-prob delta.**
   `(old_log_probs - new_log_probs).mean()` is only a loose proxy for KL. For stability you might compute an analytic KL for Normals (mean/std) or at least use `F.kl_div` over logits. If you keep the proxy, consider clipping ratios *before* KL to avoid spikes.

5. **`select_action` assumes lag features are embedded in the tail of `obs_tensor`.**
   In your training path you pass lag features separately; for consistency, either always pass them separately or always concatenate them into the observation.

6. **Advantage/return arrays mix tensor/list types.**
   Buffers store tensors for most fields but convert `advantages`/`returns` into lists of tensors later. It works with the current stacking logic, but consistency (all tensors) will reduce shape/typing edge-cases.

7. **Monitoring threads** are daemonized and only stopped by flipping a flag. If you run lots of short-lived tests, consider a join or a context manager to ensure clean shutdowns.

8. **Confidence calc: duplicated dead code.**
   In `calculate_confidence`, the `except` block returns `0.5`, but then there’s a second, nearly identical confidence computation **after** that `return` (unreachable).
   **Fix:** delete the duplicated tail or move it into the main try path.

9. **Training-step sampling under `torch.no_grad()`.**
   `_record_training_step` samples actions within `no_grad()` (fine for data collection), but note you’re saving sampled **actions/log\_probs/values**; gradients are only used in the update phase.

10. **Explained variance calculation.**
    You compute it at episode end using numpy; that’s fine. Make sure `returns_np.var()` can’t be \~0 to avoid division spikes (you guard with `>1e-6`, good).

---

want me to turn those fixes into minimal diffs you can paste in, or capture any other module next?
# Module Master Ledger

## PPOLagAgent (meta)

**ID**: `PPOLagAgent` • **Version**: `3.0.0` • **Path**: `modules/meta/ppo_lag_agent.py`
**Status**: Production-ready • **Runtime**: PyTorch + SmartInfoBus • **Owner**: *TBD*

**Description**
Advanced PPO agent with market-aware lag features, dual critics, adaptive entropy/LR, and SmartInfoBus integration. Tracks positions/risk and adapts to regimes.

**Provides (SmartInfoBus keys)**

* `agent_status`
* `ppo_lag_training_metrics` *(namespaced)*
* `position_metrics`
* `market_adaptation`

**Requires (SmartInfoBus keys)**

* `trades`, `actions`, `market_data`, `training_signals`

**Core Classes**

* `PPOLagAgent` (module)
* `MarketAwarePPONetwork` (actor-critic with attention, dual value heads)

**Hot Paths / APIs**

* `async process(**inputs)` — full tick: buffers → train step → adapt → metrics → bus
* `async propose_action(**inputs)` — action + confidence/value estimate
* `select_action(obs_tensor)` — market-aware sampling for tensors
* `record_step(...), end_episode()` — legacy async wrappers
* `get_state()/set_state()` — persistence

**Configuration (dataclass `PPOLagConfig`)**

* Core: `obs_size=64`, `act_size=2`, `hidden_size=128`, `lr=1e-4`, `lag_window=20`
* PPO: `clip_eps=0.1`, `gae_lambda=0.95`, `gamma=0.99`, `ppo_epochs=4`, `batch_size=64`, `target_kl=0.01`, `max_grad_norm=0.5`
* Features: `adv_decay=0.95`, `vol_scaling=True`, `position_aware=True`, `device="cpu"`
* Ops: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_episode_length=10`

**Adaptation / Safety**

* Regime presets: `trending/volatile/ranging/unknown` → clip/σ/LR adjustments
* Dual critics (conservative `min(v1,v2)`)
* NaN-safe inputs/outputs; action std clamps; return clamps

**Buffers & Features**

* Lag buffers: price returns, volume, spread, volatility (`lag_window` × 4)
* Extended obs: base obs + lag features + 6 position features (if `position_aware`)

**Telemetry / Health**

* Performance: `PerformanceTracker` metric `agent_cycle`
* Health: `_health_status` with NaN param checks & reward heuristics
* Background monitor thread: 30s cadence (`_start_monitoring()`)

**Circuit Breaker**

* States: `CLOSED/OPEN` • Threshold: `3` consecutive failures

**SmartInfoBus Writes (namespaced to avoid collisions)**

* `agent_status`, `ppo_lag_training_metrics`, `position_metrics`, `market_adaptation` (+ `_thesis` string)

**Known Notes**

* Training metrics are **namespaced** (`ppo_lag_training_metrics`) — do not rely on global `training_metrics`.
* Skip BatchNorm on batch size 1 to avoid instability.

---

## EnhancedWorldModel (models)

**ID**: `EnhancedWorldModel` • **Version**: `4.1.0` • **Path**: `modules/models/world_model.py`
**Status**: Production-ready • **Runtime**: PyTorch LSTM + Attention + SmartInfoBus • **Owner**: *TBD*

**Description**
Sequence world model for market simulation/prediction. LSTM backbone, multi-head attention, context fusion, multi-head outputs (price/vol/regime/confidence), scenario generation, and comprehensive analytics.

**Provides (SmartInfoBus keys)**

* `market_predictions`
* `scenario_generation`
* `world_model_analytics`
* `prediction_confidence`

**Requires (SmartInfoBus keys)**

* `market_data` (canonical). Soft/optional: `market_conditions`, `time_risk_analysis`, `trading_data`, `performance_metrics`, etc.

**Operational Modes (`WorldModelMode`)**
`INITIALIZATION`, `DATA_COLLECTION`, `TRAINING`, `CALIBRATION`, `ACTIVE_PREDICTION`, `SCENARIO_GENERATION`, `OPTIMIZATION`, `MAINTENANCE`, `ERROR_RECOVERY`

**Hot Paths / APIs**

* `async process(**inputs)` — ingest → features → predict → train (as needed) → scenarios → analytics → bus
* `async propose_action(**inputs)` / `propose_action_legacy()` — 4-asset action vector from predictions
* `get_observation_components()` — 12-dim status vector for downstream agents
* Legacy helpers: `step()`, `fit_on_history()`, `simulate_scenarios()`, `fit()`, `simulate()`

**Configuration (`WorldModelConfig`)**

* Net: `input_size=16`, `hidden_size=64`, `num_layers=2`, `dropout=0.1`, `attention_heads=4`
* Train: `learning_rate=1e-3`, `weight_decay=1e-5`, `gradient_clip=1.0`, `batch_size=64`
* Seq: `sequence_length=50`, `prediction_horizon=10`, `scenario_steps=20`
* Data: `history_size=1000`, `min_training_samples=100`, `validation_split=0.2`
* Ops: `max_processing_time_ms=200`, `circuit_breaker_threshold=5`, `min_prediction_quality=0.6`, `min_training_quality=0.7`
* Device/Perf: `device="auto"`, `use_mixed_precision=False`, `compile_model=False`, `deterministic=False`
* Monitoring: `health_check_interval=60`, `performance_window=100`, `confidence_threshold=0.5`
* Artifacts: `save_dir="artifacts/models/world_model"`

**Architecture**

* LSTM (uni, dropout if `num_layers>1`) → Multi-Head Attention → Context encoder (16-feat) → Fusion
* Heads:

  * `price_head` → 4 price changes
  * `volatility_head` → 4 vol predictions
  * `regime_head` → 4-class softmax
  * `confidence_head` → scalar \[0,1]
* AMP (CUDA only, guarded) & AdamW + ReduceLROnPlateau

**Data & Features**

* Feature set (16): normalized prices/changes, regime/session one-hots, volatility level, risk snapshot, trading/perf scalars
* Histories: `market_history`, `feature_history`, `prediction_history` (windowed), `training_history`
* Scenario cache with summaries (returns, vols, drawdowns, regime distro, quality)

**Adaptation / Training**

* Auto-train triggers: sufficiency, quality decay, elapsed time, cadence
* Early stopping, gradient clipping, LR scheduling
* Confidence blends prediction/attention entropy

**Telemetry / Health**

* Performance: `PerformanceTracker` metric `world_model_processing`
* Health monitor hook (`HealthMonitor.record`)
* Background monitor thread: interval = `health_check_interval`

**Circuit Breaker**

* States: `CLOSED/OPEN/HALF_OPEN` • Threshold: `5` • Auto half-open after reset window

**SmartInfoBus Writes**

* `market_predictions` (mode, confidence, latest preds + confidence level)
* `scenario_generation` (availability, scenarios, parameters)
* `world_model_analytics` (architecture, perf, data status, curves)
* `prediction_confidence` (scores, classification, recommendations)
* `_thesis` — compact operator summary

**Known Notes**

* Determinism guarded (seeds + cudnn flags) when `deterministic=True`.
* AMP scaler only instantiated on CUDA to avoid deprecation warnings.
* Namespaced reads prefer canonical keys; legacy fallbacks retained for compatibility.

---

### Cross-Module Integration

* The **WorldModel** publishes `market_predictions` and `prediction_confidence`; **PPOLagAgent** can consume these via its own `market_data/context` and integrate via `get_observation_components()`.
* Both modules emit operator-readable `_thesis` strings and write namespaced metrics to SmartInfoBus to avoid key collisions.
got it — here’s a clean **ledger entry** you can paste into `modules.md` for the file you just shared.

---

# RAW CAPTURE — PositionManager

**File:** `modules/position/position.py`
**Class:** `PositionManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator (@module):**

* `name="PositionManager"`, `version="3.2.0"`, `category="position"`
* `provides = ['balance','current_pnl','current_positions','equity','execution_data','order_data','portfolio_state','position_health','recent_trades','trades']`
* `requires = ['market_data','market_context','market_conditions','market_regime','price_data','prices','indicators','technical_indicators','volatility_data','instrument_signals','portfolio_metrics','market_state','environment_config','risk_score','time_risk_analysis','correlation_matrix','liquidity_capabilities','liquidity_score','market_liquidity']`
* `thesis_required=True`, `explainable=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`, `is_voting_member=True`

## What it does (1‑liner)

Hierarchical **position management**: builds a canonical market snapshot (bus-first, inputs-overlay), makes per‑instrument decisions, sizes risk, publishes portfolio & execution feeds, and proposes actions.

## Inputs (canonical + aliases)

* **Prices**: `price_data` (preferred), `prices` (fallback)
* **Indicators**: `indicators` (preferred), `technical_indicators` (fallback)
* **Volatility**: `volatility_data` (e.g., ATR)
* **Signals**: `instrument_signals` (preferred bus), *inputs fallbacks*: `signals` / `alpha_signals` / `action_signals` / `trading_signals`
* **Context/Regime**: `market_context`, `market_conditions`, `market_regime`, `trading_session` (via context)
* **Risk**: `time_risk_analysis` (preferred), `risk_score` (legacy adapter), `correlation_matrix`
* **Liquidity**: `liquidity_score` (preferred), adapters: `liquidity_capabilities` (per-instrument or global), `market_liquidity` (legacy)
* **Portfolio**: `portfolio_metrics`, `market_state`, `environment_config` (initial balance)
* **Runtime kwargs** (non‑bus): `config` (TradingConfig|dict), `instruments`, `genome`, `env`

## Outputs (bus writes)

* Per step (and on error/fallback too):

  * `trades: List[dict]` (snapshot; last 20 also in `recent_trades`)
  * `balance: float`, `equity: float`, `current_pnl: float`
  * `execution_data: dict`, `order_data: dict`
  * `portfolio_state: {health_score, exposure_ratio, open_positions, decision_quality}`
  * `position_health: {portfolio_health, exposure_ratio, risk_management_score, consecutive_losses}`
  * `current_positions: Dict[str, dict]`
  * `position_decision_<INSTRUMENT>`: `{decision, intensity, size, confidence, risk_factors}`
* Returns (function output): rich dict mirroring the above plus `position_decisions`, `position_analysis`, `_thesis`, `processing_time_ms`.

## Optional inputs / internal fallbacks

* If no `instrument_signals`, derives **intensity** from indicators: trend (SMA20/50), MACD, RSI, scaled by vol.
* If `liquidity_score` absent: adapts from `liquidity_capabilities` (global/per‑instrument) or `market_liquidity`.
* If portfolio/balance absent: adapts via `market_state` → `portfolio_metrics` → `env` → `environment_config`.

## Side‑effects

* **SmartInfoBus**: heavy writer of accounting/health/decisions keys (see “Outputs”).
* **Logging**: shared `RotatingLogger` at `logs/position/position.log` (plain‑english operator mode), immediate writes (async off).
* **Monitoring**: background thread `_start_monitoring()` updates `position_health` every \~30s.
* **Performance**: records `processing_time_ms` to `PerformanceTracker`.
* **Live broker**: optional sync/close via `env.broker` or MetaTrader5 if present.

## Voting?

* `is_voting_member=True` (publishes per‑instrument `position_decision_*` intensities and exposes `propose_action()`).

## Explainable?

* Yes. Generates `_thesis` per step; all bus writes include a thesis string. `thesis_required=True`.

## Timeout / Priority

* None declared; inherits orchestrator defaults. Runs in main step and also has a monitoring daemon thread.

## Key methods (map)

* Pipeline: `process()` → `_extract_market_data_from_inputs()` + `_extract_market_data_from_smartbus()` → `_merge_market_maps()` → `process_market_signals()` → `_update_smartbus_with_decisions()` → `_generate_position_thesis()`
* Strategy layers: `_assess_portfolio_health()`, `_assess_market_regime()`, `_extract_signal_context()`, `_make_position_decision()`
* Sizing & risk: `calculate_size()`, `_assess_risk_factors()`
* Actions: `propose_action()`, `calculate_confidence()`, `confidence()`
* Ops/health: `_apply_exit_rules()`, `_sync_live_positions()`, `_update_position_health()`, `_adapt_parameters()`
* State: `get_state()`, `set_state()`, `reset()`; evolution: `set_genome()`, `mutate()`, `crossover()`

## Internal state

* Positions & tracking: `open_positions`, `last_decisions`, `position_confidence`, `_position_metadata`, `_position_performance`, `_exit_signals`
* Histories: `_decision_history(≤100)`, `_portfolio_health_history(≤50)`, `_exposure_history(≤100)`, `signal_history[inst]`
* Scores: `_portfolio_health_score`, `_total_exposure_ratio`, `_decision_quality_score`, `_risk_management_score`
* Adaptive params: `dynamic_max_pct`, `signal_sensitivity`, `risk_tolerance`, `confidence_threshold`
* Circuit breaker dict for PM ops
* Config mirrors: typed `self.C` + dict `self.config` (kept in sync)

## Error paths & fallbacks

* Any failure in `process()` or “no market data” → publishes **empty‑but‑valid** bus feeds (providers still exist), returns safe payload with thesis.
* Guards around broker sync/close, parameter adaptation, health update; logs warnings not crashes.
* Legacy guards for `risk_score`/signals; `np.nan_to_num` on sizing inputs.
* Threading safeguarded; monitoring loop is daemonized.

## Consumers (typical)

* **Risk**: DrawdownRescue, DynamicRiskController, TimeAwareRiskScaling (reads `position_health`, `portfolio_state`, `balance/equity/current_pnl`).
* **Auditing/Explain**: AuditingCoordinator, TradeExplanationAuditor (reads `trades`, `portfolio_state`, thesis).
* **Execution Quality**: ExecutionQualityMonitor (reads `execution_data`, `order_data`).
* **Strategy/Voting**: downstream analytics might read `position_decision_*` for telemetry.
* **Replay/Analytics**: HistoricalReplayAnalyzer (reads `trades`, `current_positions`).

## Owners of its inputs (expected)

* `price_data/prices` → MarketDataProvider (or your market feed)
* `indicators/technical_indicators`, `volatility_data` → Feature engines (e.g., Advanced/MultiScale FE)
* `instrument_signals` → Ensemble/Strategy Arbiter (single canonical owner)
* `market_context/conditions/regime/session` → SessionManager / Regime detectors
* `time_risk_analysis` / `liquidity_score` → Time‑risk module / LiquidityHeatmapLayer
* `portfolio_metrics` / `market_state` → Portfolio/Risk modules or Env adaptor

## Conflicts / Duplicate‑owner risks (watchlist)

* `trades`, `recent_trades`, `balance`, `equity`, `current_pnl` **must be single‑writer**. If any external “accounting” module also publishes these, decide one canonical owner (usually PositionManager or a dedicated Portfolio service).
* `instrument_signals`: PositionManager **reads only**. Ensure PM is **not** republishing any global `trading_signals` key (it doesn’t).
* `liquidity_score`: if both LiquidityHeatmapLayer and another risk module publish it, reconcile to a single owner (PM will adapt but duplicates cause race conditions).

## Test hooks (quick stubs)

* **No-data path**: `process({})` returns safe payload + bus feeds exist.
* **Minimal map**: feed `price_data`, `indicators`, `volatility_data` for one instrument; expect a non‑hold decision if intensity derives > threshold.
* **Bus-first merge**: publish `instrument_signals` on bus and pass conflicting inputs; expect **bus intensity wins** after merge.
* **Health loop**: after 35s, `position_health` should update on bus.
* **Live sync**: mock `env.broker.get_positions()` → `open_positions` reflects broker; closing triggers logs and bus updates.

Here’s the **ledger entry for ActiveTradeMonitor**, written in the same style as the `modules.md` you’re building:

---

# RAW CAPTURE — ActiveTradeMonitor

**File:** `modules/risk/active_trade_monitor.py`
**Class:** `ActiveTradeMonitor(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name = "ActiveTradeMonitor"`
* `version = "3.1.0"`
* `category = "risk"`
* `provides = ["position_duration_risk", "duration_alerts", "position_tracking"]`
* `requires = ["positions", "market_context"]`
* `description = "Enhanced position duration monitoring with intelligent risk assessment"`
* `thesis_required = True`
* `health_monitoring = True`
* `performance_tracking = True`
* `error_handling = True`

**Contract Guarantees:**

* Always returns `position_duration_risk`, `duration_alerts`, `position_tracking`, `_thesis`
* Single-writer: only writes its owned keys (`position_duration_risk`, `duration_alerts`, `position_tracking`)
* All timestamps ISO-8601, numpy scalars cast to Python types
* Background health monitor daemon + circuit breaker for repeated errors

**Ledger fields:**

* **Category:** Risk
* **Provides:** `position_duration_risk`, `duration_alerts`, `position_tracking`
* **Requires:** `positions`, `market_context`
* **Optional Inputs:** `unrealised_pnl` / `pnl` / `entry_step` / `duration` / `bars_held` / `step_idx`
* **Outputs (shape):**

  * `position_duration_risk`: `{risk_score: float, severity_level: str, monitoring_results: {...}, risk_metrics: {...}, timestamp: str}`
  * `duration_alerts`: `{critical: [..], warning: [..], info: [..]}`
  * `position_tracking`: `{durations: {id:int}, velocities: {id:int}, statistics: {...}}`
  * `_thesis`: plain-English string
* **Side-effects:**

  * Writes status/health under namespaced keys: `active_trade_monitor_status`, `active_trade_monitor_health`
  * Writes only its owned provides keys to SmartInfoBus
  * Logs to `logs/risk/active_trade_monitor.log`
* **Voting?** No
* **Explainable?** Yes — generates `_thesis` and recommendations
* **Timeout/Priority:**

  * Circuit breaker trips if ≥4 failures in 10 cycles; resets after 20s cooldown
  * Health monitor loop: every \~5s
* **Key Methods:**

  * `process()` — main async loop, contract payloads
  * `_monitor_positions_comprehensive()` — duration + velocity analysis
  * `_assess_position_severity_enhanced()` — thresholds adjusted by regime/volatility context
  * `_calculate_comprehensive_risk_metrics()` — alert/concentration/velocity risk combined
  * `_generate_monitoring_thesis()` — plain-English rationale
  * `propose_action()` — structured risk recommendations
  * Background: `_start_monitoring()` → namespaced health/status feeds
* **State:**

  * Tracks per-position duration, velocity, first\_seen timestamps
  * Maintains closure analytics (`normal`, `timeout`, `emergency`)
  * Regime performance registry with duration history
* **Error Paths:**

  * `_generate_error_response()` → fallback contract payload with `severity_level=error`
  * Circuit breaker → `_fallback_payload()` with breaker state info
* **Consumers of its outputs:** TBD (when more modules ledgered)
* **Owners of its inputs:** TBD (positions & market\_context producers not yet mapped)
* **Conflicts/Duplicates:** None (single-writer discipline enforced)
Here’s the **ledger entry** for `EnhancedAnomalyDetector`, in the same contract-tight style:

---

# RAW CAPTURE — EnhancedAnomalyDetector

**File:** `modules/risk/anomaly_detector.py`
**Class:** `EnhancedAnomalyDetector(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name = "EnhancedAnomalyDetector"`
* `version = "4.1.0"`
* `category = "risk"`
* `provides = ["anomaly_detection", "anomaly_score", "anomaly_alerts", "detection_analytics"]`
* `requires = ["risk_data", "market_data", "trading_data", "performance_data"]`
* `thesis_required = True`
* `health_monitoring = True`
* `performance_tracking = True`
* `error_handling = True`
* `is_voting_member = True`

**Contract guarantees:**

* Always returns all `provides` keys + `_thesis` (success, fallback, or error).
* Writes **only its provides keys** to SmartInfoBus (health/status are namespaced).
* Timestamps = ISO-8601, numpy scalars → Python types.
* Circuit breaker (threshold=5) with cooldown reset, background monitor updates health.

**Ledger fields:**

* **Category:** Risk

* **Provides:** `anomaly_detection`, `anomaly_score`, `anomaly_alerts`, `detection_analytics`

* **Requires:** `risk_data`, `market_data`, `trading_data`, `performance_data`

* **Optional Inputs:** `pnl`, `volume`, `price`, `observation/obs`, `trades`

* **Outputs (shape):**

  * `anomaly_detection`: `{current_mode, enabled, anomaly_score:float, detection_confidence:float, total_anomalies:int, training_mode:bool, training_progress:float, is_training_complete:bool, timestamp:str}`
  * `anomaly_score`: `{anomaly_score:float, detection_confidence:float, anomaly_types:dict, critical_anomalies:int, emergency_mode:bool}`
  * `anomaly_alerts`: `{emergency_mode:bool, critical_anomalies_present:bool, high_anomaly_score:bool, low_detection_quality:bool, circuit_breaker_open:bool, recent_anomalies:dict}`
  * `detection_analytics`: `{detection_quality:float, detection_effectiveness:[..], threshold_adaptation_count:int, current_thresholds:dict, base_thresholds:dict, detection_stats:dict, data_sufficiency:dict, performance_metrics:dict}`
  * `_thesis`: plain-English rationale string

* **Side-effects:**

  * SmartInfoBus writes: only `anomaly_detection`, `anomaly_score`, `anomaly_alerts`, `detection_analytics`.
  * Namespaced health/status: `anomaly_detector_status`, `anomaly_detector_health`.
  * Logging: `logs/risk/enhanced_anomaly_detector.log`.

* **Voting?** Yes (`is_voting_member=True`), outputs can influence ensemble risk votes.

* **Explainable?** Yes — `_thesis` generated each cycle.

* **Timeout/Priority:**

  * Circuit breaker opens after ≥5 consecutive failures; cooldown = 20s.
  * Background health monitor loop: every 30s.

* **Key Methods:**

  * `process()` — main async contract-safe loop.
  * `_extract_detection_data()` — consolidates risk/market/trading/performance snapshots.
  * `_detect_*_anomalies_async()` — PnL, volume, price, observation, volatility, system, market structure.
  * `_analyze_patterns_async()` — sequence, correlation, trade pattern analyzers.
  * `_adapt_thresholds_async()`, `_calculate_comprehensive_score_async()`, `_update_training_progress_async()`.
  * `_handle_emergency_situations_async()`, `_update_operational_mode_async()`.
  * `_generate_detection_thesis()` — plain-English summary.
  * `propose_action()`, `calculate_confidence()` — contract-tight overrides for voting integration.

* **State:**

  * Histories: pnl/volume/price/observation/volatility (deques).
  * Baselines per regime/session/volatility.
  * Buckets: anomalies by type.
  * Metrics: anomaly\_score, detection\_confidence, detection\_quality, detection\_stats.
  * Training: progress, adaptive thresholds, threshold history.
  * Circuit breaker state.
  * Mode: `AnomalyDetectionMode` (initialization, training, calibration, active, enhanced, emergency, maintenance).

* **Error paths:**

  * `_handle_disabled_fallback()` → safe snapshot with disabled mode.
  * `_handle_no_data_fallback()` → maintains state, decays confidence.
  * `_handle_detection_error()` → logs, opens breaker if needed, pessimistic fallback payload.

* **Consumers of its outputs:** TBD (will be filled once more modules ledgered).

* **Owners of its inputs:** TBD (to be mapped after full MD ledger).

* **Conflicts/Duplicates:** None detected (single-writer discipline for its provides).

