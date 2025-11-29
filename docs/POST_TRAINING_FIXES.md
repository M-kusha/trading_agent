# Post-Training Fixes

## Investigation Date: November 28, 2025
## Training Status: Episode 255, Step ~230,000/1,000,000 (23%)

This document captures all issues discovered during training investigation that should be fixed **after training completes** to avoid disrupting the learning process.

---

## ✅ FIXED DURING TRAINING (Critical)

### Memory Veto Deadlock - FIXED

**File:** `modules/memory/components/mistakes.py`

**Problem:** Once `consecutive_losses >= 5`, veto was permanent with no recovery mechanism:
- Veto blocks all new trades
- No trades means no wins
- No wins means `consecutive_losses` never resets
- **DEADLOCK!**

**Fix Applied:**
- Added `_veto_start_time` to track when veto triggered
- Added `_veto_timeout_seconds = 60.0` (veto expires after 60 seconds)
- After timeout, `consecutive_losses` decays by 2 (allows recovery)
- Timer resets on any winning trade

---

## 🔴 HIGH PRIORITY FIXES - ✅ ALL FIXED

### 1. ThemeExpert Not Publishing When Neutral - ✅ FIXED

**File:** `modules/voting/experts/theme.py`

**Fix Applied:** Added SmartInfoBus publishing to `_neutral_output()` method. Now publishes even when returning neutral/flat votes.

**Problem:** When `_neutral_output()` is called (insufficient data, errors), it returns data but **does NOT publish to SmartInfoBus**. Only the normal path publishes.

**Evidence:** `ThemeExpert_voting_proposal` stale for ~1 hour in system.log

**Root Cause:** Lines 767-795 - `_neutral_output()` only returns dict, never calls `smart_bus.set()`

**Fix Required:**
```python
def _neutral_output(self, reason: str) -> Dict[str, Any]:
    """Generate neutral output with explanation."""
    proposal = "flat"
    confidence = 0.1
    thesis = f"Theme flat: {reason}"
    
    # ADD THIS BLOCK - Publish even when neutral
    try:
        self.smart_bus.set('ThemeExpert_voting_proposal', proposal, 
                          module=self.module_name, thesis=thesis)
        self.smart_bus.set('ThemeExpert_confidence', confidence, 
                          module=self.module_name, thesis=f'Confidence: {confidence:.1%}')
        self.smart_bus.set('theme_voting_proposal', proposal, 
                          module=self.module_name, thesis=thesis)
        self.smart_bus.set('theme_confidence', confidence, 
                          module=self.module_name, thesis=f'Theme confidence: {confidence:.1%}')
    except Exception:
        pass
    
    return { ... existing code ... }
```

---

### 2. TradingModeManager False "Slow Operation" Warnings - ✅ FIXED

**File:** `modules/trading_modes/trading_mode.py`

**Fix Applied:** Removed the `record_metric` calls for `mode_duration` and `mode_persistence` since these are tick counters, not millisecond durations. The performance tracker was interpreting them as slow operations.
2. **Rename to avoid confusion:**
```python
# Don't use record_metric for counters - it expects duration_ms
# Just log it or publish to bus directly
self.smart_bus.set('mode_duration_ticks', 
                   self.mode_stats['current_mode_duration'],
                   module='TradingModeManager',
                   thesis='Ticks in current trading mode')
```

---

## 🟡 MEDIUM PRIORITY FIXES

### 3. Trade Count Discrepancy (Executor vs UI)

**Problem:** Executor shows 156 total trades, UI shows 34 trades

**Evidence:**
- Executor log: `Total Trades: 156`
- UI display: `Trades: 34 │ W/L: 16/18`
- InfoBus `closed_positions`: 34 entries

**Root Cause:** Executor's `closed_positions` list accumulates across ALL episodes (trimmed to 500), but UI displays from InfoBus which only has current episode data. This is actually **correct behavior** - not a bug.

**No Fix Needed:** This is expected. The 156 trades are cumulative across 255 episodes, while 34 is the current episode count. The win rate (47.1%) is calculated per-episode which is correct for training.

**Optional Enhancement:** Add episode-scoped trade tracking to executor for clearer logging:
```python
# In Executor.__init__
self.episode_trades: int = 0
self.total_trades: int = 0

# In process(), detect episode reset:
if self.step_idx < self._last_step_idx:  # Step went backward = new episode
    self.episode_trades = 0
    self.closed_positions.clear()  # Optional: clear for cleaner episode stats
```

---

### 4. SessionManager Stale During Training

**File:** `modules/external/session.py` (if exists)

**Problem:** `session_canonical` key stale for ~1 hour during training

**Root Cause:** SessionManager only updates when actual trading sessions change (Tokyo→London→NewYork). During offline training with historical data, sessions don't change.

**Impact:** Low - session info is informational, not critical for training

**Fix Options:**
1. **Accept as-is** - session data isn't used during training
2. Add synthetic session progression based on data timestamps:
```python
# In SessionManager.process()
if execution_mode == "simulation":
    # Derive session from current bar timestamp
    current_time = market_data.get('timestamp', datetime.now())
    session = self._derive_session_from_time(current_time)
    self.smart_bus.set('session_canonical', session, ...)
```

---

## 🟢 LOW PRIORITY / INFORMATIONAL

### 5. UnifiedMemory Slow Process Cycle (880ms)

**File:** `modules/memory/unified.py`

**Evidence:** `performance.log` shows `process_cycle: 880ms`

**Analysis:** This appears to be legitimate heavy processing, not a bug. Memory modules do significant work:
- Pattern matching
- Playbook recall
- Danger zone detection
- Mistake tracking

**Recommendation:** Profile after training to identify optimization opportunities. Consider:
- Async processing
- Caching expensive computations
- Reducing lookback windows during training

---

### 6. Duplicate InfoBus Keys (Case Sensitivity)

**Evidence:** PowerShell error parsing JSON:
```
Cannot convert the JSON string because a dictionary that was converted from 
the string contains the duplicated keys 'position_decision_eurusd' and 'position_decision_EURUSD'
```

**Root Cause:** Modules publishing with inconsistent case:
- `position_decision_eurusd` (lowercase)
- `position_decision_EURUSD` (uppercase)

**Fix:** Standardize all instrument keys to uppercase:
```python
instrument = instrument.upper()  # Before publishing
key = f"position_decision_{instrument.upper()}"
```

---

## Summary Statistics

| Category | Count | Priority |
|----------|-------|----------|
| High Priority Fixes | 2 | Must fix after training |
| Medium Priority | 2 | Nice to have |
| Low Priority | 2 | Future optimization |

## Training Metrics Snapshot

- **Progress:** 23% (230k/1M steps)
- **Episodes:** 255 completed
- **Mean Reward:** +237.94 (up from -30 at start)
- **Balance:** €3,178.78 (+€178.78 profit)
- **Win Rate:** 47.1% (current episode)
- **Trade Bias:** 2.3:1 Long preference
- **Instrument Preference:** XAU/USD (Gold) over EUR/USD

---

*Document created for post-training implementation. Do NOT apply these fixes while training is in progress.*
