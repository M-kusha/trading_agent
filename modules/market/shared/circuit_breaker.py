

import random
import threading
import time
from collections import deque
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:


    DEFAULT_THRESHOLD = 5
    DEFAULT_WINDOW_SECONDS = 30.0
    DEFAULT_MIN_CALLS = 20
    DEFAULT_FAIL_RATE = 0.5
    DEFAULT_OPEN_BASE = 15.0
    DEFAULT_OPEN_MAX = 120.0
    DEFAULT_HALF_OPEN_MAX_CALLS = 1
    DEFAULT_HALF_OPEN_SUCCESSES = 1

    def __init__(
        self,
        name: str,
        threshold: int = DEFAULT_THRESHOLD,
        timeout: float = DEFAULT_OPEN_BASE,
        recovery_timeout: float = DEFAULT_OPEN_BASE,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        min_calls: int = DEFAULT_MIN_CALLS,
        failure_rate_threshold: float = DEFAULT_FAIL_RATE,
        open_base_timeout: float = DEFAULT_OPEN_BASE,
        open_max_timeout: float = DEFAULT_OPEN_MAX,
        half_open_max_calls: int = DEFAULT_HALF_OPEN_MAX_CALLS,
        half_open_successes_to_close: int = DEFAULT_HALF_OPEN_SUCCESSES,
        jitter_fraction: float = 0.1,
    ):

        self.name = name


        self.threshold = int(threshold)

        self.open_base_timeout = float(open_base_timeout if open_base_timeout is not None else timeout or recovery_timeout)
        self.open_max_timeout = float(open_max_timeout)

        self.window_seconds = float(window_seconds)
        self.min_calls = int(min_calls)
        self.failure_rate_threshold = float(failure_rate_threshold)


        self.half_open_max_calls = int(half_open_max_calls)
        self.half_open_successes_to_close = int(half_open_successes_to_close)


        self.jitter_fraction = float(max(0.0, min(jitter_fraction, 0.5)))


        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_streak = 0
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
        self.total_failures = 0
        self.total_successes = 0


        self._calls_window = deque()
        self._fails_window = deque()


        self._half_open_in_flight = 0
        self._half_open_successes = 0


        self._opened_at = 0.0
        self._open_until = 0.0


        self.state_changes = []
        self.max_failures = 0
        self._last_transition_reason = "init"
        self._state_entered_at = time.time()
        self._state_durations = {s.value: 0.0 for s in CircuitState}


        self._lock = threading.RLock()


    def allow_request(self) -> bool:
        now = time.time()
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:

                if now >= self._open_until:
                    self._transition_to(CircuitState.HALF_OPEN, reason="open_window_elapsed")

                else:
                    return False


            if self._half_open_in_flight < self.half_open_max_calls:
                self._half_open_in_flight += 1
                return True
            return False

    def record_success(self):
        now = time.time()
        with self._lock:
            self.total_successes += 1
            self.last_success_time = now
            self._push_call(now)

            if self.state == CircuitState.CLOSED:

                self.failure_count = 0
                self.failure_streak = 0
                return

            if self.state == CircuitState.HALF_OPEN:

                self._half_open_successes += 1

                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

                if self._half_open_successes >= self.half_open_successes_to_close:

                    self._transition_to(CircuitState.CLOSED, reason="half_open_success_threshold")
                    self.failure_count = 0
                    self.failure_streak = 0
                    self._half_open_successes = 0
                    self._half_open_in_flight = 0


    def record_failure(self):
        now = time.time()
        with self._lock:
            self.total_failures += 1
            self.last_failure_time = now
            self.max_failures = max(self.max_failures, self.failure_count + 1)
            self._push_call(now, failed=True)


            self.failure_streak += 1

            if self.state == CircuitState.HALF_OPEN:

                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                self._half_open_successes = 0
                self._reopen(reason="half_open_probe_failed")
                return

            if self.state == CircuitState.CLOSED:

                self.failure_count += 1
                if self.failure_count >= self.threshold:
                    self._reopen(reason="consecutive_failures_threshold")
                    return


                calls, fails = self._window_counts(now)
                if calls >= self.min_calls:
                    fail_rate = fails / max(1, calls)
                    if fail_rate >= self.failure_rate_threshold:
                        self._reopen(reason=f"failure_rate_{fail_rate:.2f}_over_threshold")
                        return

    def reset(self):
        with self._lock:
            self._transition_to(CircuitState.CLOSED, reason="manual_reset")
            self.failure_count = 0
            self.failure_streak = 0
            self._half_open_in_flight = 0
            self._half_open_successes = 0
            self._opened_at = 0.0
            self._open_until = 0.0

    def get_state(self) -> str:
        with self._lock:
            return self.state.value

    def get_state_dict(self) -> Dict[str, Any]:
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

                for s in CircuitState:
                    self._state_durations[s.value] = float(durations.get(s.value, self._state_durations.get(s.value, 0.0)))

            ho = state_dict.get('half_open', {})
            self._half_open_in_flight = int(ho.get('in_flight', 0))
            self._half_open_successes = int(ho.get('successes', 0))

    def get_statistics(self) -> Dict[str, Any]:
        with self._lock:
            total = self.total_successes + self.total_failures
            self._update_current_state_duration()

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


    def protect(self, func: Callable, *args, **kwargs):
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


    def _push_call(self, ts: float, failed: bool = False):
        self._calls_window.append(ts)
        if failed:
            self._fails_window.append(ts)
        self._prune_window(ts)

    def _prune_window(self, now: float):
        cutoff = now - self.window_seconds
        dq = self._calls_window
        while dq and dq[0] < cutoff:
            dq.popleft()
        fq = self._fails_window
        while fq and fq[0] < cutoff:
            fq.popleft()

    def _window_counts(self, now: float) -> Tuple[int, int]:
        self._prune_window(now)
        return len(self._calls_window), len(self._fails_window)

    def _dynamic_open_duration(self) -> float:
        k = max(0, self.failure_streak - self.threshold)
        base = self.open_base_timeout
        duration = min(self.open_max_timeout, base * (2 ** k))
        if self.jitter_fraction > 0:
            jitter = duration * self.jitter_fraction
            duration = max(0.0, duration + random.uniform(-jitter, jitter))
        return duration

    def _reopen(self, reason: str):
        now = time.time()
        self._transition_to(CircuitState.OPEN, reason=reason)
        self._opened_at = now
        duration = self._dynamic_open_duration()
        self._open_until = now + duration

        self.failure_count = 0

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
