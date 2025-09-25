# ─────────────────────────────────────────────────────────────
# File: modules/market/shared/circuit_breaker.py
# Circuit breaker pattern for fault tolerance — Production Upgrade
# ─────────────────────────────────────────────────────────────

import time
import random
import threading
from enum import Enum
from collections import deque
from typing import Dict, Any, Optional, Callable, Awaitable, Tuple


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"         # Normal operation
    OPEN = "open"             # Blocking requests
    HALF_OPEN = "half_open"   # Limited probes to test recovery


class CircuitBreaker:
    """
    Production-grade circuit breaker with:
      • Both consecutive-failure and failure-rate tripping
      • Dynamic open duration (exponential backoff + jitter)
      • Half-open concurrency controls and success threshold
      • Thread-safe internal state
      • Rich stats and transition reasons
    """

    # -------- Defaults tuned for typical market component cadence --------
    DEFAULT_THRESHOLD = 5                # consecutive failures to trip
    DEFAULT_WINDOW_SECONDS = 30.0        # sliding window for failure-rate
    DEFAULT_MIN_CALLS = 20               # min calls before rate rule applies
    DEFAULT_FAIL_RATE = 0.5              # 50% failures in window trips
    DEFAULT_OPEN_BASE = 15.0             # base open (seconds)
    DEFAULT_OPEN_MAX = 120.0             # cap for exponential open (seconds)
    DEFAULT_HALF_OPEN_MAX_CALLS = 1      # concurrent probes
    DEFAULT_HALF_OPEN_SUCCESSES = 1      # successes needed to fully close

    def __init__(
        self,
        name: str,
        threshold: int = DEFAULT_THRESHOLD,
        timeout: float = DEFAULT_OPEN_BASE,           # kept for backward compat (open base)
        recovery_timeout: float = DEFAULT_OPEN_BASE,  # kept for backward compat (alias of base)
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        min_calls: int = DEFAULT_MIN_CALLS,
        failure_rate_threshold: float = DEFAULT_FAIL_RATE,
        open_base_timeout: float = DEFAULT_OPEN_BASE,
        open_max_timeout: float = DEFAULT_OPEN_MAX,
        half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_CALLS,
        half_open_successes_to_close: int = DEFAULT_HALF_OPEN_SUCCESSES,
        jitter_fraction: float = 0.1,                 # up to ±10% jitter on open duration
    ):
        # Identity
        self.name = name

        # Trip policies
        self.threshold = int(threshold)
        # Back-compat mapping: if caller set `timeout` or `recovery_timeout`, prefer explicit open_base_timeout
        self.open_base_timeout = float(open_base_timeout if open_base_timeout is not None else timeout or recovery_timeout)
        self.open_max_timeout = float(open_max_timeout)

        self.window_seconds = float(window_seconds)
        self.min_calls = int(min_calls)
        self.failure_rate_threshold = float(failure_rate_threshold)

        # Half-open controls
        self.half_open_max_calls = int(half_open_max_calls)
        self.half_open_successes_to_close = int(half_open_successes_to_close)

        # Jitter for open duration
        self.jitter_fraction = float(max(0.0, min(jitter_fraction, 0.5)))

        # State
        self.state = CircuitState.CLOSED
        self.failure_count = 0               # consecutive failures (CLOSED only)
        self.failure_streak = 0              # consecutive failures across all states (for backoff)
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.total_failures = 0
        self.total_successes = 0

        # Sliding window of outcomes (timestamps only for speed)
        self._calls_window = deque()         # all calls timestamps
        self._fails_window = deque()         # failed calls timestamps

        # Half-open internals
        self._half_open_in_flight = 0
        self._half_open_successes = 0

        # Open-phase timing
        self._opened_at = 0.0
        self._open_until = 0.0

        # Stats & transitions
        self.state_changes = []              # list of dicts with reason
        self.max_failures = 0
        self._last_transition_reason = "init"
        self._state_entered_at = time.time()
        self._state_durations = {s.value: 0.0 for s in CircuitState}

        # Lock for thread-safety
        self._lock = threading.RLock()

    # ----------------------------- Public API -----------------------------

    def allow_request(self) -> bool:
        """Check if the next request should be allowed."""
        now = time.time()
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                # Can we try half-open probes?
                if now >= self._open_until:
                    self._transition_to(CircuitState.HALF_OPEN, reason="open_window_elapsed")
                    # fallthrough to HALF_OPEN logic
                else:
                    return False

            # HALF_OPEN: allow up to N concurrent probes
            if self._half_open_in_flight < self.half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            return False

    def record_success(self):
        """Record a successful execution."""
        now = time.time()
        with self._lock:
            self.total_successes += 1
            self.last_success_time = now
            self._push_call(now)

            if self.state == CircuitState.CLOSED:
                # Reset consecutive failures
                self.failure_count = 0
                self.failure_streak = 0
                return

            if self.state == CircuitState.HALF_OPEN:
                # One of our probes succeeded
                self._half_open_successes += 1
                # Decrement in-flight counter (probe finished)
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

                if self._half_open_successes >= self.half_open_successes_to_close:
                    # Recovery confirmed — fully close
                    self._transition_to(CircuitState.CLOSED, reason="half_open_success_threshold")
                    self.failure_count = 0
                    self.failure_streak = 0
                    self._half_open_successes = 0
                    self._half_open_in_flight = 0
                # else remain HALF_OPEN until success threshold met or a failure occurs

    def record_failure(self):
        """Record a failed execution (counts toward tripping rules)."""
        now = time.time()
        with self._lock:
            self.total_failures += 1
            self.last_failure_time = now
            self.max_failures = max(self.max_failures, self.failure_count + 1)
            self._push_call(now, failed=True)

            # Any failure increments the global streak (affects open duration backoff)
            self.failure_streak += 1

            if self.state == CircuitState.HALF_OPEN:
                # Probe failed — immediately re-open with backoff
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes = 0
                self._reopen(reason="half_open_probe_failed")
                return

            if self.state == CircuitState.CLOSED:
                # Consecutive failure rule
                self.failure_count += 1
                if self.failure_count >= self.threshold:
                    self._reopen(reason="consecutive_failures_threshold")
                    return

                # Sliding failure-rate rule (only if enough calls)
                calls, fails = self._window_counts(now)
                if calls >= self.min_calls:
                    fail_rate = fails / max(1, calls)
                    if fail_rate >= self.failure_rate_threshold:
                        self._reopen(reason=f"failure_rate_{fail_rate:.2f}_over_threshold")
                        return

    def reset(self):
        """Reset breaker to CLOSED and clear counters (does not clear history)."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED, reason="manual_reset")
            self.failure_count = 0
            self.failure_streak = 0
            self._half_open_in_flight = 0
            self._half_open_successes = 0
            self._opened_at = 0.0
            self._open_until = 0.0

    def get_state(self) -> str:
        """Get current state as string."""
        with self._lock:
            return self.state.value

    def get_state_dict(self) -> Dict[str, Any]:
        """Get state for serialization."""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'failure_streak': self.failure_streak,
                'threshold': self.threshold,
                'total_failures': self.total_failures,
                'total_successes': self.total_successes,
                'last_failure_time': self.last_failure_time,
                'last_success_time': self.last_success_time,
                'max_failures': self.max_failures,
                'state_changes': len(self.state_changes),
                'open_until': self._open_until,
                'opened_at': self._opened_at,
                'last_transition_reason': self._last_transition_reason,
                'durations': dict(self._state_durations),
                'window_seconds': self.window_seconds,
                'min_calls': self.min_calls,
                'failure_rate_threshold': self.failure_rate_threshold,
                'half_open': {
                    'max_calls': self.half_open_max_calls,
                    'successes_to_close': self.half_open_successes_to_close,
                    'in_flight': self._half_open_in_flight,
                    'successes': self._half_open_successes,
                },
            }

    def set_state(self, state_dict: Dict[str, Any]):
        """Restore state from dictionary."""
        with self._lock:
            self.state = CircuitState(state_dict.get('state', CircuitState.CLOSED.value))
            self.failure_count = state_dict.get('failure_count', 0)
            self.failure_streak = state_dict.get('failure_streak', 0)
            self.total_failures = state_dict.get('total_failures', 0)
            self.total_successes = state_dict.get('total_successes', 0)
            self.last_failure_time = state_dict.get('last_failure_time', 0.0)
            self.last_success_time = state_dict.get('last_success_time', 0.0)
            self.max_failures = state_dict.get('max_failures', 0)
            self._open_until = state_dict.get('open_until', 0.0)
            self._opened_at = state_dict.get('opened_at', 0.0)
            self._last_transition_reason = state_dict.get('last_transition_reason', "restored")
            durations = state_dict.get('durations')
            if isinstance(durations, dict):
                # Only accept known states
                for s in CircuitState:
                    self._state_durations[s.value] = float(durations.get(s.value, self._state_durations.get(s.value, 0.0)))
            # Half-open fields
            ho = state_dict.get('half_open', {})
            self._half_open_in_flight = int(ho.get('in_flight', 0))
            self._half_open_successes = int(ho.get('successes', 0))

    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        with self._lock:
            total = self.total_successes + self.total_failures
            self._update_current_state_duration()
            # Uptime = percent time in CLOSED
            durations = dict(self._state_durations)
            total_time = sum(durations.values()) or 1.0
            uptime = (durations.get(CircuitState.CLOSED.value, 0.0) / total_time) * 100.0

            calls, fails = self._window_counts(time.time())
            return {
                'name': self.name,
                'current_state': self.state.value,
                'failure_count': self.failure_count,
                'success_rate': self.total_successes / max(total, 1),
                'failure_rate': self.total_failures / max(total, 1),
                'total_requests': total,
                'max_consecutive_failures': self.max_failures,
                'state_transitions': len(self.state_changes),
                'uptime_pct': uptime,
                'durations_sec': durations,
                'open_until': self._open_until,
                'last_transition_reason': self._last_transition_reason,
                'sliding_window': {
                    'seconds': self.window_seconds,
                    'calls': calls,
                    'failures': fails,
                    'fail_rate': (fails / max(1, calls)),
                }
            }

    # ---------------------- Optional call wrappers -----------------------

    def protect(self, func: Callable, *args, **kwargs):
        """
        Protect a synchronous call with the circuit breaker.
        Raises RuntimeError if not allowed or re-raises the function's exception.
        """
        if not self.allow_request():
            raise RuntimeError("Circuit open")
        try:
            result = func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result

    async def protect_async(self, func: Callable[..., Awaitable], *args, **kwargs):
        """
        Protect an async call with the circuit breaker.
        Raises RuntimeError if not allowed or re-raises the coroutine's exception.
        """
        if not self.allow_request():
            raise RuntimeError("Circuit open")
        try:
            result = await func(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result

    # ----------------------------- Internals -----------------------------

    def _push_call(self, ts: float, failed: bool = False):
        """Record a call in the sliding window."""
        self._calls_window.append(ts)
        if failed:
            self._fails_window.append(ts)
        self._prune_window(ts)

    def _prune_window(self, now: float):
        """Drop old timestamps outside the sliding window."""
        cutoff = now - self.window_seconds
        dq = self._calls_window
        while dq and dq[0] < cutoff:
            dq.popleft()
        fq = self._fails_window
        while fq and fq[0] < cutoff:
            fq.popleft()

    def _window_counts(self, now: float) -> Tuple[int, int]:
        """(calls, failures) within window_seconds."""
        self._prune_window(now)
        return len(self._calls_window), len(self._fails_window)

    def _dynamic_open_duration(self) -> float:
        """
        Exponential backoff on open duration based on failure_streak.
        open_time = base * 2^(k) capped to open_max, with ±jitter.
        """
        k = max(0, self.failure_streak - self.threshold)
        base = self.open_base_timeout
        duration = min(self.open_max_timeout, base * (2 ** k))
        if self.jitter_fraction > 0:
            jitter = duration * self.jitter_fraction
            duration = max(0.0, duration + random.uniform(-jitter, jitter))
        return duration

    def _reopen(self, reason: str):
        """Transition to OPEN and compute open_until with backoff."""
        now = time.time()
        self._transition_to(CircuitState.OPEN, reason=reason)
        self._opened_at = now
        duration = self._dynamic_open_duration()
        self._open_until = now + duration
        # Reset counters relevant to CLOSED state
        self.failure_count = 0
        # Half-open prep
        self._half_open_in_flight = 0
        self._half_open_successes = 0

    def _transition_to(self, new_state: CircuitState, reason: str):
        if self.state == new_state:
            return
        now = time.time()
        self._update_current_state_duration(now)
        old_state = self.state
        self.state = new_state
        self._last_transition_reason = reason
        self.state_changes.append({
            'from': old_state.value,
            'to': new_state.value,
            'timestamp': now,
            'failure_count': self.failure_count,
            'reason': reason
        })
        self._state_entered_at = now

    def _update_current_state_duration(self, now: Optional[float] = None):
        now = now or time.time()
        prev = self._state_entered_at
        if prev <= 0:
            self._state_entered_at = now
            return
        self._state_durations[self.state.value] = self._state_durations.get(self.state.value, 0.0) + max(0.0, now - prev)
        self._state_entered_at = now
