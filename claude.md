# 📒 Trading System Module Documentation

This document provides a comprehensive overview of all modules in the trading system, organized by their functional categories and locations.

---

## 🔍 modules/auditing/

### AuditingCoordinator

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

#### 1) Public API

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

#### 2) Inputs (from `requires` and code usage)

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

#### 3) Outputs (from `provides` and return)

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

#### 4) Side-effects / Bus / Logging

* Calls `InfoBusManager.get_instance()` and **writes directly** to bus once:

  * `smart_bus.set('audit_status', <status>, module=<class name>, thesis=<str>, confidence=<float>)`
* Also returns all three provided keys; orchestrator will typically publish those too (outside this file).
* Logging via `self.logger.info/error` with status and counts.

#### 5) Internal state & lifecycle

* State set in `_initialize()`:

  * `discovered_auditors: dict` (currently reset to `{}` by `_discover_audit_modules()`)
  * `audit_modules: list` (unused in current snippet)
  * `cross_validation_cache: dict`
  * `audit_session_start: datetime`
  * `audit_performance: dict` with `total_audits`, `successful_audits`, `failed_audits`, `avg_audit_time`
* `reset()` clears caches, resets timers/counters, and logs.

#### 6) Core logic (summaries)

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

#### 7) Concurrency / timing

* `process` and two helper methods are `async`.
* Metadata has `timeout_ms=150` (enforced outside this file).

#### 8) Error paths & fallbacks

* Any exception in `process`:

  * logs `[FAIL]`
  * updates performance with `success=False`
  * returns safe result with `audit_status='failed'`, `confidence=0.0`, `_thesis` describing the error.
* Any exception in `_perform_comprehensive_audit`:

  * `overall_status='failed'` and an `error` field in results.

#### 9) Voting / explainability flags

* `is_voting_member = False`
* Provides two hooks anyway:

  * `propose_action` → describes an `audit_coordination` action with target modules = keys of `discovered_auditors`.
  * `calculate_confidence` → returns confidence based on session success rate (capped to 0.9).
* `explainable = True` → `_thesis` is included in returns and also attached to bus set for `audit_status`.

#### 10) Connections (to be filled as we ingest other files)

* **Consumes:** `trading_signal`, `market_data`, `trades`
  *(owners TBD — will be filled from other module captures and/or Contracts file)*
* **Provides:** `audit_status`, `audit_report`, `audit_metrics`
  *(consumers TBD — monitoring/UX modules likely; will fill when we see them)*

#### 11) Gaps/notes (no action now)

* `_discover_audit_modules()` currently does not populate `discovered_auditors`; discovery mechanism likely handled by orchestrator or pending.
* `market_data` is required but not used inside audits (kept for future consistency checks).
* Only `audit_status` is written directly to the bus; `audit_report`/`audit_metrics` are returned but not explicitly `set()` here (likely set by orchestrator).

---

### TradeExplanationAuditor

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

#### RAW CAPTURE — TradeExplanationAuditor

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

#### 1) Public API (methods)

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

#### 2) Inputs (from `requires` + code usage)

Expected via `process(**inputs)` or derived:

* `trading_signal: dict` (passed into context; not directly scored in quality)
* `market_data: dict` (passed into context; not directly used in scoring)
* `trades: list[dict]` — iterated; each trade can include:

  * `symbol: str`
  * `timestamp: any` (used to compose `trade_id`)
  * `action: str` (quality boost if `'buy'|'sell'|'hold'`)
  * `pnl: number`
  * `confidence: float` (affects quality)
  * `reason: str` or `_thesis: str` (counts as "has thesis")
  * optional: `risk_assessment` (presence adds to quality)
* Optional extras handled:

  * `timestamp: datetime` (defaults to `datetime.now()`)
  * `step_idx: int` (defaults to `0`)
* Internal context adds: `risk_score: 0.5` (constant default in this file)

#### 3) Outputs (from `provides` + return)

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

#### 4) Side-effects / Logging

* SmartInfoBus write: **only** `trade_explanations` is set from inside the module.
* Operator logs via `self.logger.info` using `format_operator_message(...)` when:

  * `abs(pnl) > 50` **or** `explanation_quality < 0.5`.
* Errors logged with `[FAIL]` prefixes.

#### 5) Core logic & scoring

##### 5.1 Per-trade explanation object (from `_audit_single_trade_explanation`)

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

##### 5.2 Quality scoring (from `_validate_explanation_quality`)

* Start at `0.0`
* **Thesis present** (`has_thesis`): `+0.4`; else add issue `missing_thesis`
* **Confidence:**

  * `> 0.7` → `+0.3`
  * `< 0.3` → add issue `low_confidence` and `+0.1`
  * else → `+0.2`
* **Action clarity:** action in `{'buy','sell','hold'}` → `+0.2`; else add `unclear_action`
* **Risk assessment:** if `'risk_assessment' in trade` → `+0.1`; else add `missing_risk_assessment`
* Max theoretical score: **1.0**

##### 5.3 Batch analysis (from `_analyze_explanations`)

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

##### 5.4 Alerts (from `_check_for_alerts`)

If `trade_count == 0` → `[]`. Otherwise potential alerts:

* **Low explanation quality** if `avg_quality < 0.5`
  → `{type: 'low_explanation_quality', severity: 'high', ...}`
* **Low confidence pattern** if `avg_confidence < 0.4`
  → `{type: 'low_confidence_pattern', severity: 'medium', ...}`
* **Missing explanations** if `(missing_thesis_count / trade_count) > alert_thresholds['missing_explanation_rate']` (default `0.1`)
  → `{type: 'missing_explanations', severity: 'medium', ...}`
* **Risk-based** if `context['risk_score'] > 0.8` **and** `avg_confidence < 0.6`
  → `{type: 'high_risk_low_confidence', severity: 'critical', ...}`

##### 5.5 Thesis (from `_generate_explanation_thesis`)

* If no trades: `"No trades processed for explanation auditing in this cycle."`
* Else:

  * `quality > 0.8` and `confidence > 0.7` → "Excellent …"
  * `quality > 0.6` and `confidence > 0.5` → "Good …"
  * otherwise → "Needs improvement …"
  * appends alert count if any.

#### 6) Running session metrics (from `quality_metrics` + updater)

* `total_trades_audited`
* `high_confidence_trades` (+= `trade_count - low_confidence_count`)
* `low_confidence_trades`
* `missing_explanations`
* `pattern_violations`
  (+= number of trades in `detailed_trades` with `len(quality_issues) > 2`)

#### 7) Reports & statistics

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

#### 8) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Module metadata `timeout_ms=100` (enforced externally).

#### 9) Voting / explainability

* Not a voting member.
* Provides a neutral `propose_action` with `_thesis` describing focus.
* `calculate_confidence`: if `total_trades_audited == 0` → `0.5`; else
  returns `min(0.9, max(0.1, high_confidence_rate * (1 - missing_explanation_rate)))`.

#### 10) State & lifecycle

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

#### 11) Data shapes (canonical, copy-paste)

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

#### 12) Minimal example I/O

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

#### 13) Connections (to fill as we ingest more files)

* **Consumes:** `trading_signal`, `market_data`, `trades` (owners TBD)
* **Provides:** `trade_explanations`, `audit_alerts`, `explanation_metrics` (consumers TBD)

#### 14) Notes / gaps (no action now)

* `_extract_audit_context` sets a fixed `risk_score=0.5` (not sourced from bus).
* `market_regime` used in per-trade object comes from `context` but isn't populated in this file.
* `alert_thresholds['low_confidence_rate']` and `['pattern_violation_rate']` exist but are **not** used directly in `_check_for_alerts`.

---

### TradeThesisTracker

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

#### RAW CAPTURE — TradeThesisTracker

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

#### 1) Public API (methods)

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

#### 2) Inputs (from `requires` + code usage)

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

#### 3) Outputs (from `provides` + return)

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

#### 4) Side-effects / Logging

* SmartInfoBus write: `thesis_analysis` (with thesis & confidence).
* Logs thesis changes via `format_operator_message("🧠", "Thesis change: old → new", ...)`.

#### 5) Core logic

##### 5.1 Thesis extraction (`_extract_current_thesis`)

* If no `trading_signal` → returns `"no_signal"`.
* Builds `confidence_level`: `high` if `signal_confidence>0.7`, `medium` if `>0.4`, else `low`.
* Computes `market_regime` from `market_data.close` (if available) as `trending_up`/`trending_down`; defaults to `unknown`.
* Base thesis string: `f"{action}_{market_regime}_{confidence_level}"`.
* If `signal_reason` contains any of `{bullish,bearish,breakout,reversal,momentum,support,resistance}` (case-insensitive), appends first found word to thesis.

##### 5.2 Change handling (`_handle_potential_thesis_change` / `_handle_thesis_change`)

* Increments `thesis_changes` when `new_thesis != current_thesis`.
* Records transition in `transition_matrix[old][new] += 1`.
* Appends change record to `thesis_history` with timestamp and change number.
* Updates `current_thesis` and logs an operator message.

##### 5.3 Trade processing & performance

* `_process_trades_with_thesis` builds `processed_trade` objects:

  * fields: `trade_id = f"{symbol}_{timestamp}"`, `symbol`, `action` (default `'unknown'`), `pnl` (default `0`), `thesis=current_thesis`, `confidence=context.signal_confidence`, `timestamp=context.timestamp`, `processed_at=ISO`.
* `_update_thesis_performance` updates:

  * per-thesis `trades += 1`, `pnl += trade.pnl`
  * per-thesis `confidence` EMA with `alpha=0.2` (seeded by first value)
* `_update_best_worst_thesis` sets `best_thesis`/`worst_thesis` by max/min total `pnl`.

##### 5.4 Analysis & alerts

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

#### 6) Running state & thresholds

* State initialized in `_initialize`:

  * `current_thesis="unknown"`, `thesis_changes=0`
  * `thesis_performance = defaultdict(lambda: {"trades":0,"pnl":0.0,"confidence":0.0})`
  * `thesis_history = deque(maxlen=500)`
  * `thesis_patterns = defaultdict(int)` *(not incremented in this file)*
  * `transition_matrix = defaultdict(lambda: defaultdict(int))`
  * `session_start = now`, `best_thesis=None`, `worst_thesis=None`
  * `alert_thresholds = {'frequent_changes':10,'poor_performance':-100,'low_confidence':0.3}`

#### 7) Reports & statistics

* `get_comprehensive_thesis_analysis()` returns base analysis plus:

  * `transitions` (dict-of-dicts), `patterns` (dict copy), `recent_history` (last 20 changes),
  * `trading_summary` via `_get_trading_summary()` if available on mixin.
* `generate_thesis_report()` prints a formatted report including:

  * session duration, current thesis, change count,
  * current P\&L/confidence, total P\&L/trades,
  * best/worst thesis with P\&L,
  * change frequency, diversity, active thesis count.

#### 8) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Metadata `timeout_ms=100` (enforced externally).

#### 9) Voting / explainability

* Not a voting member.
* `propose_action` returns `{'action_type':'thesis_tracking', ... , '_thesis': '...'}`.
* `calculate_confidence` combines:

  * `stability_score = 1 - (thesis_changes / max(1, total_theses*10))` clamped `[0.1,1]`
  * `performance_score = (total_pnl + 1000)/2000` clamped `[0.1,1]`
  * final `confidence = (stability_score + performance_score)/2` clamped `[0.1,0.9]`.
* `explainable=True`; `_thesis` string returned and attached to bus write.

#### 10) Data shapes (canonical, copy-paste)

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

#### 11) Minimal example I/O

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

#### 12) Connections (to fill as we ingest more files)

* **Consumes:** `trading_signal`, `market_data`, `trades` (owners TBD)
* **Provides:** `thesis_analysis`, `thesis_performance`, `thesis_alerts` (consumers TBD)

#### 13) Notes / gaps

* `thesis_patterns` declared but not incremented in this file.
* Market regime detection is simplistic (close\[-1] vs close\[-5]); no smoothing or timeframe input.
* Bus write performed only for `thesis_analysis`; other provided keys are returned (publication likely handled by orchestrator).

---

## 📡 modules/external/

### MarketDataProvider

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

#### RAW CAPTURE — MarketDataProvider

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

#### 1) Public API (methods)

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

#### 2) Inputs

* No bus `requires`; reads local CSVs under `cfg.data_directory`.

#### 3) Outputs (from `provides` + `_build_snapshot` / `_empty_snapshot`)

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

#### 4) Side-effects / Logging

* Disk I/O reading CSVs in `_load_data_files`.
* Logs via `RotatingLogger("MarketDataProvider", "logs/external/market_data_provider.log")`.
* Maintains internal performance/health counters (`_success/_fail/_proc_times`).

#### 5) Core logic summary

* `_load_data_files`: scans `cfg.data_directory` for patterns like `XAUUSD_H1_features.csv` / `EURUSD_D1.csv` (case-insensitive). Requires `timestamp` or `time` column; coerces to datetime. Accepts close-only datasets: fills O/H/L from `close`, `volume=0`. Cleans NaNs/infs, sorts by timestamp, de-dupes. Builds `data_files[symbol][tf]` and `data_iterators[symbol]`.
* `_advance_symbol_data`: steps the primary TF iterator for a symbol; restarts at beginning on `StopIteration`. Builds `current_bars[symbol]` with OHLCV and optional `bid/ask`; updates buffers/indicators.
* `_update_technical_indicators`: computes SMA(20/50), RSI(14) (simple), ATR(14) from buffers.
* `process`: throttled by `cfg.update_frequency` seconds; advances all supported symbols, updates session labels, returns `_build_snapshot`; on exception returns `_empty_snapshot`.
* `calculate_confidence`: blends availability, freshness (≤60s), and basic quality checks into \[0,1].
* `propose_action`: suggests update & maintenance every 2000 updates; includes `data_quality`.

#### 6) Configuration (dataclass `MarketDataConfig`)

* `data_directory: str = "data/processed"`
* `supported_symbols: list[str] = ["XAU/USD","EUR/USD"]`
* `supported_timeframes: list[str] = ["H1","H4","D1"]`
* `primary_timeframe: str = "H4"`
* `update_frequency: float = 1.0`
* `buffer_size: int = 10000`
* `enable_technical_indicators: bool = True`

#### 7) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* Internal throttling via `update_frequency` and `_last_update_ts`.

#### 8) Connections

* **Consumes:** none (root).
* **Provides to (observed so far):** AuditingCoordinator, TradeExplanationAuditor, TradeThesisTracker via `market_data`.
* **Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by `SessionManager`.

#### 9) Notes / gaps

* If no valid CSVs found: returns schema-complete but empty snapshot; never fabricates values.
* `volatility_hint` derived from normalized ATR vs price; thresholds fixed in code.
* `historical_prices` is an alias of `multi_timeframe_data` for compatibility.

---

### SessionManager

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

#### RAW CAPTURE — SessionManager

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

#### 1) Public API (methods)

* `def __init__(self, config: Optional[Dict[str, Any]] = None)`
* `def _initialize(self) -> None`
* `def _update_session_labels(self) -> None`
* `def _session_canonical(self) -> str`
* `async def calculate_confidence(self, action: Optional[Dict[str, Any]] = None, **inputs) -> float`
* `async def propose_action(self, **inputs) -> Dict[str, Any]`
* `async def process(self, **inputs) -> Dict[str, Any]`

#### 2) Inputs

* No bus `requires`; uses internal timers/labels only.

#### 3) Outputs (from `provides` + `process`)

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

#### 4) Side-effects / Logging

* Logs via `RotatingLogger("SessionManager", "logs/external/session_manager.log")`.
* Maintains `system_alerts` (append on errors).

#### 5) Core logic summary

* `_update_session_labels`: sets human labels (`trading_session`, `session_type`) based on UTC hour.
* `_session_canonical`: maps UTC hour to `'asian'|'european'|'us'|'closed'`.
* `process`: updates labels and timestamps, refreshes `_last_health_check`, builds the snapshot (metrics, context, performance, health, alerts, labels, time\_of\_day), increments success and records proc time; on exception, increments fail, appends an error alert, returns degraded snapshot.

#### 6) Configuration (dataclass `SessionConfig`)

* `session_duration: int = 3600`
* `performance_window: int = 500`
* `enable_health_monitoring: bool = True`
* `enable_performance_tracking: bool = True`
* `enable_error_pinpointing: bool = True`

#### 7) Concurrency / timing

* `process`, `propose_action`, `calculate_confidence` are `async`.
* `calculate_confidence`: combines freshness of last health check (≤60s) and counter availability into \[0,1].
* `propose_action`: suggests `reset_session=True` when `duration >= session_duration`.

#### 8) Connections

* **Consumes:** none (root).
* **Provides:** session/context/health keys for other modules or UI.
* **Duplicates:** `trading_session`, `session_type`, `session_canonical` also provided by `MarketDataProvider`.

#### 9) Notes / gaps

* No direct SmartInfoBus `set()` calls in this file; outputs are returned for orchestrator/bus publication.
* Health/performance counters are real-only; no fabricated metrics.

---

### NewsSentimentModule

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

#### RAW CAPTURE — NewsSentimentModule

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

#### 1) Public API (methods)

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

#### 2) Configuration (dataclass `SentimentConfig`)

* `enabled: bool = False`
* `default_sentiment: float = 0.0`
* `cache_ttl: int = 60` (seconds)
* `max_retries: int = 2`
* `timeout: float = 10.0`
* Performance thresholds: `max_processing_time_ms=300`, `circuit_breaker_threshold=3`, `min_confidence=0.3`
  **Env:** `NEWS_API_KEY` (empty string → default/fallback path)

#### 3) Inputs (from bus + optional `symbol`)

* **Bus reads** (inside `_extract_sentiment_data`):

  * `market_data` (dict)
  * `symbols` (list or scalar)
  * `trading_session` (str/dict; used only for context)
* `symbol` (optional in `process(**inputs)`): if absent, picks first from `symbols`; else from `market_data` fallback `'EURUSD'`.

#### 4) Outputs (declared via return of `process`)

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

#### 5) Core pipeline (inside `process`)

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

#### 6) Scoring & models

* **Sentiment value**: float in \[-1, 1]; random synth in `_make_api_request` (simulated API)
* **Confidence**: `0.5 + |sentiment| * 0.5` (simulated); default `0.0` when disabled/no API
* **Trend**: sign and magnitude of slope over last-5 sentiments (`np.polyfit`)

#### 7) Monitoring, health & circuit breaker

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

#### 8) Caching

* `_cache: {symbol: {sentiment, confidence, timestamp}}`
* TTL: `genome["cache_ttl"]` seconds
* Eviction: `_cleanup_expired_cache()` (monitor thread) or on stale check inside `_get_cached_sentiment`

#### 9) Logging & performance

* `RotatingLogger(..., operator_mode=True, plain_english=True)`
* Operator messages via `format_operator_message`
* Performance via `PerformanceTracker.record_metric('NewsSentimentModule', 'sentiment_analysis', ...)`

#### 10) State (persistence)

`get_state()` / `set_state()` cover:

* `latest_sentiment`, `sentiment_confidence`, `_cache`, `genome`, `_api_call_count`, `_api_failures`,
* `_sentiment_performance`, `circuit_breaker`, `_health_status`

#### 11) Data shapes (canonical, copy-paste)

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

#### 12) Connections

* **Consumes:** `market_data`, `symbols` → **MarketDataProvider**; `trading_session` → **MarketDataProvider**/**SessionManager** (duplicate providers)
* **Provides:** sentiment keys for downstream strategy/risk/monitoring modules (not shown in current files)

#### 13) Notes / gaps

* **Shape divergence** between returned outputs and direct bus writes for `news_sentiment`, `sentiment_confidence`, `sentiment_trend` (see conflicts above).
* `_sentiment_history` is referenced for trend but **never appended** in this file; trend may stay `'insufficient_data'` unless history is updated elsewhere.
* Monitoring thread (`_start_monitoring`) **not started in init/process**; only invoked in `set_sentiment()` (manual path).

---

## 🔧 modules/features/

### AdvancedFeatureEngine

**File:** `modules/features/advanced_feature_engine.py`
**Category:** features
**Provides:** `advanced_features`, `features` (alias), `feature_analysis`, `feature_thesis` (+ returns hidden `_thesis`)
**Requires:** `price_data` (hard requirement), with fallbacks via InfoBus: `historical_prices` → `ohlcv_data` → `market_data`
**Optional direct inputs supported:** `prices`, `price`, `close`, `price_series`
**Side-effects:** Writes declared keys to SmartInfoBus; background health/perf monitors (async tasks)
**Voting?** `is_voting_member=False`
**Explainable?** `thesis_required=True`
**Timeout/Priority:** `timeout_ms=120`, hot reload enabled

#### RAW CAPTURE — AdvancedFeatureEngine

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
   * Fallbacks (from InfoBus, in order): `historical_prices` (uses `close` arrays or `current_bar.close`) → `ohlcv_data` → `market_data` (keys containing "price" list or nested `{close}`).
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

### MultiScaleFeatureEngine

**File:** `modules/features/multiscale_feature_engine.py`
**Category:** features
**Provides:** `multiscale_features`, `neural_embeddings`, `attention_weights`, `feature_fusion` (+ returns hidden `_thesis`)
**Requires:** `advanced_features` (prefers InfoBus; can use injected AFE instance)
**Optional upstream keys consumed (if present):** `advanced_features_{TF}` for TF in cfg.timeframes (e.g., `advanced_features_H1`)
**Side-effects:** Writes declared keys to SmartInfoBus; GPU usage optional; background neural/gpu monitors
**Voting?** `is_voting_member=False`
**Explainable?** `thesis_required=True`
**Timeout/Priority:** `timeout_ms=180`, hot reload enabled

#### RAW CAPTURE — MultiScaleFeatureEngine

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

#### Data-flow links (between the two)

* **MarketDataProvider → AdvancedFeatureEngine**: `price_data` / `historical_prices` / `ohlcv_data` / `market_data`
* **AdvancedFeatureEngine → MultiScaleFeatureEngine**: `advanced_features.raw_features` length defines `input_dim`; optional per-TF keys `advanced_features_{TF}` enable correlations & attention over TFs.
* **Both publish to InfoBus** with **single-writer discipline** over their own keys.

---

## 📈 modules/market/

### FractalRegimeConfirmation

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
**Conflicts/Duplicates:** none on keys; overlaps conceptually with other "regime/session" providers but uses distinct keys

#### RAW CAPTURE — FractalRegimeConfirmation

**Class:** `FractalRegimeConfirmation(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin)`
**Decorator metadata (@module):**

* `name="FractalRegimeConfirmation"`, `version="3.1.0"`, `category="market"`
* `provides=["market_regime","regime_strength","trend_direction","fractal_metrics","regime_data","symbols","timestamps"]`
* `requires=["prices","step_idx","volatility_level"]`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`
* `is_voting_member=False`, `hot_reload=True`, `timeout_ms=180`

#### 1) Public API

* `async process(**inputs) -> dict` — main loop (cb gate → data extract → metrics → hysteresis → bus → return)
* `step(...) -> (label, strength)` — legacy sync step (no bus writes)
* `propose_action(...)`, `calculate_confidence(...)`, `confidence(...)`
* Reporting/obs: `get_observation_components()`, `get_regime_analysis_report()`

#### 2) Inputs the code accepts

* **Preferred structured (legacy):** `data_dict` (map `{symbol: {TF: DataFrame}}`) + `current_step` (+ optional `theme_detector`)
* **Current InfoBus path:** `prices` (dict of last prices), `step_idx` (int), `volatility_level` (str)
* Fallback: last known `prices` cache or synthetic generator (if nothing available)

#### 3) Outputs (contract)

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

#### 4) Side-effects / Bus writes

* Publishes **declared keys only**: `market_regime`, `regime_strength`, `trend_direction`, `regime_data`, `symbols`, `timestamps` (with operator-style `thesis` strings).

#### 5) Core logic

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

#### 6) Health / circuit breaker

* CB states: `CLOSED` → `OPEN` at ≥3 consecutive failures → `HALF_OPEN` after 120s; on success resets.
* On CB-open or data-missing: returns cached state + minimal metrics; logs via `ErrorPinpointer` and `EnglishExplainer`.

#### 7) Notable details / gaps

* `requires` lists `step_idx`/`volatility_level`, but extraction mostly relies on InfoBus reads; direct `inputs` path is the legacy `data_dict/current_step`.
* `fractal_metrics` filter in `_format_declared_outputs` keeps only finite numeric values.
* Theme integration hook (`theme_detector`) adjusts strength (0.5–1.0 multiplier) but defaults to `1.0`.

---

### LiquidityHeatmapLayer

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

#### RAW CAPTURE — LiquidityHeatmapLayer

**Class:** `LiquidityHeatmapLayer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator metadata (@module):**

* `name="LiquidityHeatmapLayer"`, `version="3.0.2"`, `category="market"`
* `provides=["liquidity_score","market_depth","spread_analysis","liquidity_prediction","trading_sessions","session_data","liquidity_thesis","liquidity_capabilities"]`
* `requires=["bid_ask_data","price_data","prices"]`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

#### 1) Public API

* `async process(**inputs) -> dict` (cb gate → data extract → analytics → NN prediction → sessions → bus → return)
* `propose_action(...)` (recommends trade posture & size), `calculate_confidence(...)`
* Health/Reports: `get_health_status()`, `get_liquidity_performance_report()`
* State I/O: `get_state()/set_state()`

#### 2) Inputs the code accepts

* **InfoBus:** `prices`, `price_data`, `bid_ask_data`

  * ⚠️ Canonicalization: it converts `"EUR/USD"` → `"EURUSD"` when querying dictionaries. Upstream should match this for hits.
* **Optional direct:** `market_data` dict containing arrays for `prices`, `volumes`, `bid_ask_spreads`, and `market_depth {bids,asks}`.
* **Synthetic fallback** if nothing present.

#### 3) Outputs (contract)

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

#### 4) Side-effects / Bus writes

* Writes **all provided keys** (score, depth, spread, prediction, sessions, session\_data, thesis, capabilities) with human theses.

#### 5) Core logic

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

#### 6) Sessions snapshot

* Derives active session from UTC (asian/european/american/rollover/weekend) and provides a `liquidity_bias` hint.

#### 7) Health / circuit breaker

* Neural CB: `OPEN` after ≥3 failures; `HALF_OPEN` after 120s; success closes CB.
* Health monitor adjusts `data_quality_score` and `model_health_score`; anomaly checker warns on extreme spreads, low depth, CB open.

#### 8) Notable details / gaps

* **Symbol keys:** Expects `"EURUSD"/"XAUUSD"` codes when querying InfoBus dicts; make sure upstream uses those (or extend matcher).
* Predictions depend on **untrained** network weights unless you wire in a training loop — treat outputs as diagnostic unless trained.
* `_format_declared_outputs` always returns `liquidity_prediction` as a dict; safe even on errors.

---

### MarketThemeDetector

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
**Circuit breakers:** ML training breaker (OPEN after N failures; reset after 300s idle) + soft "circuit\_breaker\_failures" metric
**Consumers (TBD):** regime/portfolio overlay, UI dashboards, risk posture
**Input owners (likely):** market data provider / orchestrator writing `historical_prices` or `multi_timeframe_data` and `macro_data`
**Conflicts/Duplicates:** avoids collisions—only writes theme-namespaced keys. It *includes* `market_data`/`price_data`/`technical_indicators` inside its **returned payload** (for dashboards), but **does not** publish those keys to the bus.

#### RAW CAPTURE — MarketThemeDetector

**Class:** `MarketThemeDetector(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusVotingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):**

* `name="MarketThemeDetector"`, `version="3.1.0"`, `category="market"`
* **provides:** `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`, `theme_analysis`, `theme_detection`, `theme_detector_status`, `theme_detector_health`, `theme_model_quality`
* **requires:** `market_data`, `price_data`, `technical_indicators`, `historical_prices`, `multi_timeframe_data`, `macro_data`
* `thesis_required=True`, `health_monitoring=True`, `performance_tracking=True`, `error_handling=True`

#### 1) Config (`ThemeDetectorConfig`)

* `n_themes=4`, `window=100`, `batch_size>=64` (auto: `max(64, n_themes*16)`), `feature_lookback=500`
* `instruments`: defaults to `["XAU/USD", "EUR/USD"]`
* ML: `max_iter=100`, `convergence_threshold=0.001`, `clustering_quality_threshold=0.30`
* Perf: `max_processing_time_ms=200`, `circuit_breaker_threshold=3`
* Features: `use_macro=True`, `timeframes=("H1","H4","D1")`

#### 2) Lifecycle & systems

* Initializes SmartInfoBus, rotating logger, `ErrorPinpointer`, `EnglishExplainer`, `PerformanceTracker`.
* ML stack: `StandardScaler` + `MiniBatchKMeans(n_clusters=n_themes, batch_size, max_iter, n_init=10)`.
* Background monitoring **thread** (daemon) every \~30s: updates health metrics & resets ML breaker after 300s.
* Publishes an initial `theme_detector_status`.

#### 3) Data extraction (robust, multi-path)

Order of precedence:

1. `historical_prices` **or** `multi_timeframe_data` (multi-TF dict) → used directly.
2. `market_data` (compact current-bar dict) → shaped into `{instrument: {H4: {...}}}` with arrays and `current_bar`.
3. Individually keyed: `market_data_{instrument}_{timeframe}` for configured instruments×TFs.
4. Direct `inputs["market_data"]` if passed to `process`.
5. **Last known** cached market data (warn).
6. **Synthetic** generator (deterministic-ish) for configured instruments×TFs.

#### 4) Feature engineering

For **each** instrument×timeframe, appends 7 stats (uses closing prices):

* `vol` (stdev of last 20 returns), `mom` (mean last 5 returns), `hurst` (safe est on last 50),
  `wave` (detail energy on last 30 via pywt db4), `trend` (SMA10 vs SMA30),
  `roll_10` (last/\[-10] − 1), `bars_available` (count).
  If `use_macro`: appends **3 macro** features `[vix, yield_curve, cpi]`, scaled via a separate scaler (seeded with `[20, 0.5, 3]`).
* Feature vector is padded/trimmed to a **fixed length**:
  `expected = len(instruments) * len(timeframes) * 7 + (3 if use_macro else 0)`.

#### 5) Model fitting & readiness

* A rolling **fit buffer** (`deque`, maxlen 2000) collects recent feature vectors.
* `_should_fit_model()` → **fit** when `len(buffer) >= batch_size` **and** `(ml_fit_count % 10 == 0)`.
  (i.e., fits at counts 0, 10, 20… to throttle.)
* Fit pipeline: `X` → scale → `MiniBatchKMeans.fit(X_scaled)`; increments `ml_fit_count`.
  Computes **quality** = `1 / (1 + inertia/n_samples)` (clamped 0..1).
  Publishes `theme_model_quality` to the bus.
* **Readiness**: cluster centers exist **and** `quality > clustering_quality_threshold` (default 0.30).

#### 6) Detection & scores

* If **ready**:

  * `theme_id = argmin_k distance_k(kmeans.transform(fs))` where `fs=scaled(features)`.
  * `theme_strength = 1 / (1 + min_distance)` (→ 0..1).
  * `theme_confidence` \~ **separation**: `(second_smallest − min) / second_smallest` (0..1).
* Else (cold start): defaults → `theme_id=0`, `strength≈0.30`, `confidence≈0.10`.
* Tracks histories: `_theme_vec` one-hot with strength, `_theme_momentum` (last strengths), `_theme_history`.

**Stability & transitions**

* `theme_stability`: `1 − std(last 10 strengths)` (clipped 0..1; fallback 0.5 if <5 points).
* `transition_probability`: from last 3 strengths' **momentum**: `prob = clip(-avg_diff + 0.10, 0..1)`.

#### 7) Outputs (contract & payload)

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

#### 8) SmartInfoBus writes (side-effects)

* `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`
* `theme_detection` (compact payload)
* `theme_analysis` (snapshot incl. thesis)
* `theme_detector_status` (init) and `theme_detector_health` (periodic)
* `theme_model_quality` (on each successful fit)
* **Does not** publish `market_data`/`price_data`/`technical_indicators` to avoid namespace clashes.

#### 9) Thesis generation

* Named themes:
  `0 Risk-Off Defensive`, `1 Growth Momentum`, `2 Volatility Spike`, `3 Range-Bound Consolidation` (defaults to `Theme {id}` for others).
* Human summary includes: Strength/Confidence/Stability (with labels), instruments analyzed count, clustering quality, feature stability, transition-risk notice, fit count & buffer size, timestamp.

#### 10) Health, errors & breakers

* **ML circuit breaker:** tracks failures during fit; `OPEN` after `threshold` (default 3); auto-reset to `CLOSED` after 300s.
* General error path returns **safe fallback** with `theme_detection` filled and explanatory thesis; logs via `ErrorPinpointer` + `EnglishExplainer`.
* Background health snapshot publishes `theme_detector_health`: success rate, avg processing time, breaker state, quality, data extraction rate, timestamp.

#### 11) Actions & confidence (helpers)

* `propose_action()` maps theme → action/risk:

  * **0 Risk-Off:** `reduce_exposure` (if conf ≥0.5) else `monitor` (risk: medium)
  * **1 Momentum:** `buy_moderate` or `buy_aggressive` (risk: medium; scaled by confidence)
  * **2 Vol Spike:** `defensive` (risk: high)
  * **3 Range-Bound:** `range_trade` (if conf ≥0.6) else `monitor` (risk: low)
* `action_confidence` = blend of conf/strength/stability/quality (bounded 0..1).
* `calculate_confidence()` mixes: conf, clustering quality, stability, ML breaker health, data extraction success; adjusts for aggressive calls & alignment with range-trade.

#### 12) State I/O

* `get_state()` returns current theme, vector, transitions count, quality, fit count, success/failure counts, config subset, last update.
* `set_state()` safely restores those and logs success.

---

### RegimePerformanceMatrix

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

#### RAW CAPTURE — RegimePerformanceMatrix

**Class:** `RegimePerformanceMatrix(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="RegimePerformanceMatrix"`, `version="3.1.0"`, with *provides*/*requires* as above.

#### 1) Config (`RegimeMatrixConfig`)

* `n_regimes=3`, `decay_factor=0.95`, `vol_history_size=500`, `performance_window=100`, `regime_sensitivity=1.0`
* Perf thresholds: `max_processing_time_ms=150`, `circuit_breaker_threshold=3`, `accuracy_threshold=0.60`
* Instruments for vol fallback: `("XAU/USD", "EUR/USD")`

#### 2) State & structures

* `matrix`: **n×n** float32 (exponentially-decayed P\&L by *predicted*→*true* regime)
* `volatility_regimes`: 3 anchors initial `[0.1, 0.3, 0.5]` → adapt from history percentiles
* Pointers: `_current_regime`, `_predicted_regime`, `last_volatility`, `last_liquidity`
* Histories: `vol_history`, `_performance_history`, `_regime_history`, `_predicted_regime_history`, `_true_regime_history`
* Per-regime: `_regime_accuracy_scores[n]`, `_regime_pnl_tracking[i]->deque`, `_regime_transitions` map `"i->j"`→stats
* Characteristics per regime: `avg_volatility`, `avg_pnl`, `count`, `accuracy`, `stability_score`
* Monitoring counters: `processing_times`, `success_count`, `failure_count`, `circuit_breaker_failures`

#### 3) Data extraction

`_extract_performance_data()` builds:

* **Predicted regime** from `market_regime` (accepts str/int). Mapping for strings:
  `{"trending": 0, "volatile": 1, "ranging": 2}` *(anything else → 0)*
* **Volatility**: from `volatility_data` or `_calculate_volatility_fallback()` using `market_data` **multi-TF** `{close:[...]}` (std of last ≤20 returns).
* **PnL**: `pnl_data` numeric, else sum of `recent_trades[].pnl` (if list).
* **Liquidity**: `liquidity_score` (default 1.0).
  Returns a compact dict with timestamp + source.

> ⚠️ Heads-up: regime string mapping (`trending/volatile/ranging`) may not align with other modules that use `noise/volatile/trending`. Unknown strings → 0.

#### 4) Core logic

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

#### 5) Outputs (returned payload)

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

#### 6) SmartInfoBus writes

* Performance & analytics: `regime_performance`, `regime_accuracy`, `regime_prediction`, `regime_data`, `regime_analysis`, `regime_matrix_analysis`
* State & metrics: `market_state`, `performance_metrics`, `backtesting_data`, `stress_test_results`
* Passthrough: `recent_trades` (reads then re-sets)
* Signals: `trading_signals`
* Health & status (via monitor / init): `regime_matrix_health`, `regime_matrix_status`
* **Note:** explicitly avoids writing `market_regime` (to preserve single-writer policy).

#### 7) Thesis

`_generate_matrix_thesis(...)` → compact narrative with:

* True vs Predicted regime (named: 0 Low Vol, 1 Medium Vol, 2 High Vol), prediction status, overall accuracy
* Average performance, current volatility, trend, decay factor
* Per-regime characteristics (obs, avg vol, avg PnL, accuracy)
* Accuracy tier blurb, plus quick stats: transition count, performance window fill, vol history fill

#### 8) Monitoring, fallbacks & errors

* **Monitor thread (30s):** `_update_health_metrics()` + `_update_regime_accuracy()`; publishes `regime_matrix_health`.
* **No data:** returns safe matrix & metrics with thesis; logs warning.
* **Errors:** logs via `ErrorPinpointer`/`EnglishExplainer`, increments failure counters, then falls back.

#### 9) Actions & confidence (helpers)

* `propose_action()` (uses current regime, predicted, vol, overall accuracy):

  * Regime 0 (Low vol): `"buy"` if `pred==curr` and acc>0.7 else `"hold"` (risk low)
  * Regime 1 (Med vol): `"trade"` if acc>0.6 else `"reduce_exposure"` (risk medium)
  * Regime 2 (High vol): `"defensive"` if vol>0.3 else `"cautious_trade"` (risk high)
  * `regime_confidence = min(1, overall_accuracy + (1 - vol)*0.3)`
* `calculate_confidence(action)` weighted blend:
  `0.40*overall_accuracy + 0.25*regime_stability + 0.15*data_quality + 0.10*perf_consistency + 0.10*matrix_coverage`, with action-specific tweaks.

#### 10) State I/O

* `get_state()` returns matrix, pointers, anchors, characteristics, transitions, accuracy scores, counters, config slice.
* `set_state()` restores shapes safely; logs success.

#### Quick integration notes / gotchas

* **Regime label mapping:** Reads `market_regime` strings as `trending→0`, `volatile→1`, `ranging→2`. If your upstream uses `noise` instead of `ranging`, this will default to **0**. Consider normalizing upstream values or extend the mapping.
* **recent\_trades writer:** This module **re-publishes** `recent_trades`. If other modules write to the same key, coordinate single-writer or designate a canonical owner.
* **Volatility scale assumptions:** Signal confidence uses `(1 - volatility)`. Works best when vol ∈ \~\[0,1]; if your vol is larger, consider scaling before publishing to the bus.
* **Stress scenarios:** Predefined in `_initialize_stress_testing()` but only surfaced via `stress_test_results`; no automatic execution path shown here.

---

### TimeAwareRiskScaling

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

#### RAW CAPTURE — TimeAwareRiskScaling

**Class:** `TimeAwareRiskScaling(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin)`
**Decorator (@module):** `name="TimeAwareRiskScaling"`, `version="3.1.0"` with provides/requires as above.

#### 1) Config (`TimeAwareRiskConfig`)

* **Session ends (UTC):** `asian_end=8`, `euro_end=16`, `us_end=22`
* **Risk scaling:** `base_factor=1.0`, `decay_factor=0.9`, `vol_window=100`, `session_memory=24`
* **Session multipliers:** asian `1.2`, european `1.0`, US `1.1`, closed `0.5`
* **Thresholds:** `max_processing_time_ms=100`, `circuit_breaker_threshold=3`, `risk_threshold_high=0.8`, `risk_threshold_critical=0.95`
* **Volatility fallback instruments:** `("XAU/USD", "EUR/USD")`

#### 2) State & structures

* **Per-hour arrays (length 24):** `vol_profile`, `risk_profile`, `_hourly_risk_scores`
* **Session tracking:** `_current_session`, `_session_changes`, `_session_transitions` (deque)
* **Risk tracking:** `_volatility_history` (size `vol_window`), `_factor_history`, `_risk_events`
* **Session stats:** `_session_performance[session] = {count, total_factor, avg_volatility, risk_events, success_rate, last_update}`
* **Dynamic multipliers:** `_session_risk_multipliers` (auto-tuned by success rate after ≥10 obs)

#### 3) Data extraction

* `_extract_time_data()`

  * **timestamp:** from bus `timestamp` (str/ts/dt) else now; **hour** = UTC hour
  * **volatility:** `_extract_volatility_data()` uses `volatility_data` numeric; else **expects** `market_data[instrument]["close"]` (simple series, not multi-TF) to compute std of recent returns; else falls back to `current_volatility` or `0.01`
  * **session:** `_get_session(hour)` via config cutoffs
  * pulls passthrough `market_data`, `risk_data`

#### 4) Core logic

* `_process_time_aware_scaling(time_data)`

  * Session transition bookkeeping (+ logging) and adaptive multiplier tweak
  * **Base factor:** `base_factor` decayed by recent vol ratio; mild hourly normalization via `vol_profile`
  * **Vol adjustment:** z-score on vol history → returns {1.5, 1.2, 1.0, 0.85, 0.7} bands
  * **Session multiplier:** from config, auto-adjusted by performance (`success_rate>0.8 → ×0.95`, `<0.6 → ×1.05`, clipped 0.3..2.0)
  * **Final scaling factor:** `clip(base*volAdj*sessionMult, 0.1, 5.0)`
  * **Risk level (0..1):** `0.4*factor_risk(scaling/2)` + `0.4*vol_risk(percentile of vol history)` + `0.2*session_risk_map` (asian .3, eur .2, us .25, closed .1)
  * Updates per-hour profiles; computes `risk_trend` (slope on last 5 factors), `volatility_trend` (last 10 vols), `volatility_regime` (percentile bands), `session_efficiency` (success\_rate penalized by risk\_events)

#### 5) Outputs (returned payload)

* **Declared provides:**

  * `risk_scaling_factor` (float)
  * `session_risk` `{current_session, risk_level, session_multiplier, hour}`
  * `volatility_adjustment` `{adjustment_factor, current_volatility, volatility_regime, volatility_trend}`
  * `market_conditions` `{session, hour, volatility_regime, risk_trend}`
  * `time_risk_analysis` rich blob (risk/vol trends, hourly patterns, transitions, success flag, last\_update)
  * `time_risk_status` `{status, current_session, hour, scaling_factor, risk_level, volatility, last_update}`
  * `time_risk_health` `{success_rate, avg_processing_time_ms, circuit_breaker_failures, current_risk_level, session_transitions, risk_events, last_update}`
* **Also returns (convenience):** `volatility_data` (float), `risk_data` (compact dict), `thesis`, `_thesis`

#### 6) SmartInfoBus writes

* On **process**: `risk_scaling_factor`, `session_risk`, `volatility_adjustment`, `market_conditions`, `time_risk_analysis`
* On **init**: `time_risk_status`
* On **monitor** (30s): `time_risk_health`
* Perf metric recorded each run

#### 7) Thesis

* `_generate_risk_thesis(...)` → narrative with session name, hour (UTC), scaling factor, risk level, current vol + regime, component breakdown, per-session guidance for **Gold (XAU/USD)** & **EUR/USD**, regime warnings, threshold notices, and session performance summary.

#### 8) Monitoring, thresholds & fallbacks

* **Monitor loop (30s):** updates health, analyzes session patterns every ≥300s (`_hourly_risk_scores`), checks thresholds → pushes `_risk_events` & logs `[ALERT]` if `current_risk_level` exceeds `risk_threshold_high/critical`
* **No data / errors:** safe fallback via `_format_declared_outputs` with thesis; errors annotated via `ErrorPinpointer`/`EnglishExplainer`

#### 9) Actions & confidence

* `propose_action()` calls `process()` and maps `risk_level` → `{reduce_exposure, moderate_caution, maintain_current, increase_exposure}` with `magnitude` and `confidence = min(0.9, scaling_factor/2)`
* `calculate_confidence(action)` = blend of base (history aware), scaling proximity to 1.0, risk bucket; +10% if session known

#### 10) State I/O

* `get_state()` returns session markers, profiles, performance, multipliers, counters, and key config fields
* `set_state()` restores safely; logs success
* `stop_monitoring()` flips flag (daemon exits naturally)

#### Quick integration notes / gotchas

* **Market data shape for vol fallback:** expects `market_data[INSTR]["close"]` directly under the instrument (not the multi-TF `{TF:{close:[…]}}` shape other modules use). If you publish multi-TF, consider adding a simple snapshot alongside for this module.
* **UTC assumption:** session mapping uses UTC hour; ensure your `timestamp` is UTC or adjust ends accordingly.
* **Double processing in `propose_action()`:** it calls `process()` again, which re-writes bus keys. If you chain actions frequently, consider passing cached results to avoid redundant writes.
* **Scaling → confidence coupling:** confidence caps at `0.9` and scales with factor/2; very large factors will not exceed that cap by design.
* **Threshold alerts:** only log & enqueue in `_risk_events`; there's no external notifier inside this module. If you need paging, subscribe downstream to `time_risk_health` or `time_risk_analysis`.

---

## 📊 Master Ledger Summary

| **Module** | **File** | **Category** | **Provides** | **Requires** | **Voting?** | **Explainable?** | **Timeout/Priority** | **Key Methods** | **Consumers** | **Owners** | **Conflicts** |
|------------|----------|--------------|---------------|---------------|-------------|------------------|---------------------|-----------------|-------------|-----------|-------------|
| **AuditingCoordinator** | `modules/auditing/auditing_coordinator.py` | auditing | `audit_status`, `audit_report`, `audit_metrics` | `trading_signal`, `market_data`, `trades` | No | Yes | 150ms/3 | `process`, `_audit_trading_signals`, `_audit_trades` | TBD | TBD | None |
| **TradeExplanationAuditor** | `modules/auditing/trade_explanation_auditor.py` | auditing | `trade_explanations`, `audit_alerts`, `explanation_metrics` | `trading_signal`, `market_data`, `trades` | No | Yes | 100ms/3 | `process`, `_validate_explanation_quality` | TBD | TBD | None |
| **TradeThesisTracker** | `modules/auditing/trade_thesis_tracker.py` | auditing | `thesis_analysis`, `thesis_performance`, `thesis_alerts` | `trading_signal`, `market_data`, `trades` | No | Yes | 100ms/3 | `process`, `_extract_current_thesis` | TBD | TBD | None |
| **MarketDataProvider** | `modules/external/market_data_provider.py` | external | `market_data`, `price_data`, `ohlcv_data`, `bid_ask_data`, `prices`, `symbols`, `timestamp`, `technical_indicators`, `indicators`, `volatility_data`, `volatility`, `volatility_index`, `multi_timeframe_data`, `historical_prices`, `market_context`, `trading_session`, `session_type`, `session_canonical`, `data_provider_health` | None | No | No | N/A | `process`, `_load_data_files`, `_advance_symbol_data` | Auditing modules, Feature engines, Market modules | Root | Session keys duplicate with SessionManager |
| **SessionManager** | `modules/external/session_manager.py` | external | `session_metrics`, `session_context`, `system_performance`, `system_health`, `system_alerts`, `trading_session`, `session_type`, `session_canonical`, `time_of_day` | None | No | No | N/A | `process`, `_update_session_labels` | TBD | Root | Session keys duplicate with MarketDataProvider |
| **NewsSentimentModule** | `modules/external/news_sentiment.py` | external | `news_sentiment`, `sentiment_confidence`, `news_summary`, `sentiment_trend` | `market_data`, `symbols`, `trading_session` | N/A | Yes | N/A | `process`, `_process_sentiment_analysis` | TBD | MarketDataProvider, SessionManager | Shape mismatch: returns vs bus writes |
| **AdvancedFeatureEngine** | `modules/features/advanced_feature_engine.py` | features | `advanced_features`, `features`, `feature_analysis`, `feature_thesis` | `price_data` | No | Yes | 120ms | `process`, `_extract_comprehensive_features` | MultiScaleFeatureEngine | MarketDataProvider | None |
| **MultiScaleFeatureEngine** | `modules/features/multiscale_feature_engine.py` | features | `multiscale_features`, `neural_embeddings`, `attention_weights`, `feature_fusion` | `advanced_features` | No | Yes | 180ms | `process`, `_neural_forward` | TBD | AdvancedFeatureEngine | Shape mismatch: returns vs bus writes |
| **FractalRegimeConfirmation** | `modules/market/fractal_regime_confirmation.py` | market | `market_regime`, `regime_strength`, `trend_direction`, `fractal_metrics`, `regime_data`, `symbols`, `timestamps` | `prices`, `step_idx`, `volatility_level` | No | Yes | 180ms | `process`, `_compute_fractal_metrics_robust` | TBD | MarketDataProvider | None |
| **LiquidityHeatmapLayer** | `modules/market/liquidity_heatmap_layer.py` | market | `liquidity_score`, `market_depth`, `spread_analysis`, `liquidity_prediction`, `trading_sessions`, `session_data`, `liquidity_thesis`, `liquidity_capabilities` | `bid_ask_data`, `price_data`, `prices` | No | Yes | N/A | `process`, `_neural_liquidity_prediction` | TBD | MarketDataProvider | Symbol canonicalization mismatch |
| **MarketThemeDetector** | `modules/market/market_theme_detector.py` | market | `market_theme`, `theme_strength`, `theme_confidence`, `theme_transition`, `theme_analysis`, `theme_detection`, `theme_detector_status`, `theme_detector_health`, `theme_model_quality` | `market_data`, `price_data`, `technical_indicators`, `historical_prices`, `multi_timeframe_data`, `macro_data` | No | Yes | N/A | `process`, `_detect_current_theme` | TBD | MarketDataProvider | None |
| **RegimePerformanceMatrix** | `modules/market/regime_performance_matrix.py` | market | `regime_accuracy`, `regime_prediction`, `market_state`, `performance_metrics`, `stress_test_results`, `backtesting_data`, `recent_trades`, `trading_signals`, `regime_analysis`, `regime_data`, `regime_matrix_analysis`, `regime_matrix_health`, `regime_matrix_status`, `regime_performance` | `market_regime`, `market_data`, `liquidity_score`, `recent_trades`, `volatility_data`, `pnl_data` | No | Yes | N/A | `process`, `_process_regime_matrix` | TBD | FractalRegimeConfirmation, LiquidityHeatmapLayer | Regime mapping mismatch; recent_trades re-publisher |
| **TimeAwareRiskScaling** | `modules/market/time_aware_risk_scaling.py` | risk | `risk_scaling_factor`, `session_risk`, `volatility_adjustment`, `market_conditions`, `time_risk_analysis`, `time_risk_status`, `time_risk_health` | `timestamp`, `market_data`, `risk_data`, `volatility_data` | No | Yes | N/A | `process`, `_process_time_aware_scaling` | TBD | MarketDataProvider | Market data shape expectations |

---

## 🔍 Data Flow Dependencies

### Primary Data Providers (Root)
- **MarketDataProvider** → Feeds most other modules with market data, prices, OHLCV, technical indicators
- **SessionManager** → Provides session timing and health context

### Feature Processing Chain
- **MarketDataProvider** → **AdvancedFeatureEngine** → **MultiScaleFeatureEngine**

### Market Analysis Chain
- **MarketDataProvider** → **FractalRegimeConfirmation** → **RegimePerformanceMatrix**
- **MarketDataProvider** → **LiquidityHeatmapLayer**
- **MarketDataProvider** → **MarketThemeDetector**

### Risk & Auditing Consumers
- **TimeAwareRiskScaling** ← **MarketDataProvider**
- **AuditingCoordinator** ← various trading modules
- **TradeExplanationAuditor** ← various trading modules  
- **TradeThesisTracker** ← various trading modules

### External Data Integration
- **NewsSentimentModule** ← **MarketDataProvider**, **SessionManager**

---

## ⚠️ Known Integration Issues

### 1. Duplicate Providers
- **Session keys conflict**: `trading_session`, `session_type`, `session_canonical` provided by both MarketDataProvider and SessionManager

### 2. Shape Mismatches
- **NewsSentimentModule**: Returns float for `news_sentiment` but writes dict to bus
- **MultiScaleFeatureEngine**: Returns list for `neural_embeddings` but writes dict to bus

### 3. Symbol Canonicalization
- **LiquidityHeatmapLayer** expects `"EURUSD"/"XAUUSD"` format
- Other modules may use `"EUR/USD"/"XAU/USD"` format

### 4. Regime Label Mappings
- **RegimePerformanceMatrix** maps `trending→0, volatile→1, ranging→2`
- **FractalRegimeConfirmation** uses `noise/volatile/trending`
- Potential mismatch for `ranging` vs `noise`

### 5. Market Data Shape Expectations  
- **TimeAwareRiskScaling** expects simple `market_data[instrument]["close"]`
- Other modules expect multi-timeframe structure `{TF: {close: [...]}}`

---

This comprehensive documentation provides a complete overview of all modules in the trading system, organized by category and including detailed technical specifications, data flows, and integration considerations.