

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np


class ComponentStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    DISABLED = "disabled"


@dataclass
class ComponentResult:
    component: str
    status: ComponentStatus
    data: Dict[str, Any]
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None


class BaseMarketComponent(ABC):

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        logger: Optional[Any] = None,
        metrics_tracker: Optional[Any] = None
    ):
        self.name = name
        self.config = {

            'enabled': True,
            'timeout_sec': 5.0,
            'retry_attempts': 0,
            'retry_backoff_sec': 0.25,
            'retry_backoff_max_sec': 2.0,
            'retry_jitter_sec': 0.05,

            'cb_enabled': True,
            'cb_failure_threshold': 5,
            'cb_recovery_time_sec': 15.0,
            'cb_half_open_max_calls': 1,


            'cache_enabled': False,
            'cache_ttl_sec': 2.0,
            'cache_max_entries': 128,
            'auto_cache': False,


            **(config or {})
        }
        self.logger = logger
        self.metrics_tracker = metrics_tracker


        self._initialized = False
        self._state: Dict[str, Any] = {}
        self._cache: OrderedDict[str, Tuple[float, Dict[str, Any]]] = OrderedDict()


        self._execution_count = 0
        self._success_count = 0
        self._failure_count = 0
        self._timeout_count = 0
        self._total_execution_time = 0.0
        self._last_error: Optional[str] = None


        self._cb_state = 'closed'
        self._cb_failures = 0
        self._cb_opened_at: Optional[float] = None
        self._cb_half_open_calls = 0


        self._initialize()


    def _initialize(self):
        self.trace(f"Initializing {self.name}", level="DEBUG")
        try:
            self.initialize()
            self._initialized = True
            self.trace(f"{self.name} initialized successfully", level="INFO")
        except Exception as e:
            self.trace(f"{self.name} initialization failed: {e}", level="ERROR")
            raise

    @abstractmethod
    def initialize(self):
        ...


    async def analyze(self, **inputs) -> Dict[str, Any]:
        start_time = time.time()
        self._execution_count += 1


        if not self.config.get('enabled', True):
            self.trace(f"{self.name} is disabled", level="WARNING")
            return self._decorate_fallback(
                self.get_fallback_result("Component disabled"),
                status=ComponentStatus.DISABLED,
                cache_hit=False,
                attempts=0,
                start_time=start_time
            )

        self.trace(f"{self.name}.analyze() starting", level="TRACE")
        self.trace(f"Input keys: {list(inputs.keys())}", level="TRACE")


        if self.config.get('cb_enabled', True) and not self._breaker_allows_call():
            self.trace("Circuit open: short-circuiting call", level="WARNING")
            return self._decorate_fallback(
                self.get_fallback_result("Circuit open"),
                status=ComponentStatus.CIRCUIT_OPEN,
                cache_hit=False,
                attempts=0,
                start_time=start_time
            )


        try:
            self._validate_inputs(inputs)
        except Exception as e:
            self._record_failure(e)
            return self._decorate_fallback(
                self.get_fallback_result(f"Invalid inputs: {e}"),
                status=ComponentStatus.ERROR,
                cache_hit=False,
                attempts=0,
                start_time=start_time
            )


        cache_key = self._get_cache_key(inputs) or (self._auto_cache_key(inputs) if self.config.get('auto_cache') else None)
        if cache_key and self.config.get('cache_enabled'):
            hit = self._cache_get(cache_key)
            if hit is not None:
                result = self._decorate_result(hit, start_time, cache_hit=True, attempts=0)
                self._record_success(result['_component_meta']['execution_time_ms'])
                return result


        attempts = int(self.config.get('retry_attempts', 0)) + 1
        timeout = float(self.config.get('timeout_sec', 0) or 0)
        backoff_base = float(self.config.get('retry_backoff_sec', 0.25))
        backoff_cap = float(self.config.get('retry_backoff_max_sec', 2.0))
        jitter = float(self.config.get('retry_jitter_sec', 0.05))

        last_error = None
        for attempt_idx in range(attempts):
            try:

                self._breaker_on_call_begin()

                coro = self.analyze_impl(**inputs)
                if timeout > 0:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro


                try:
                    self._validate_output(result)
                except Exception as ve:
                    raise ve


                self._breaker_on_success()


                exec_ms = (time.time() - start_time) * 1000.0
                self._record_success(exec_ms)


                if cache_key and self.config.get('cache_enabled'):
                    self._cache_set(cache_key, result)


                return self._decorate_result(result, start_time, cache_hit=False, attempts=attempt_idx + 1)

            except asyncio.TimeoutError as te:
                last_error = te
                self._timeout_count += 1
                self._breaker_on_failure()

                if attempt_idx < attempts - 1:
                    await asyncio.sleep(self._calc_backoff(attempt_idx, backoff_base, backoff_cap, jitter))
                    continue
                else:
                    self._record_failure(te)
                    return self._decorate_fallback(
                        self.get_fallback_result("Operation timed out"),
                        status=ComponentStatus.TIMEOUT,
                        cache_hit=False,
                        attempts=attempt_idx + 1,
                        start_time=start_time
                    )
            except Exception as e:
                last_error = e
                self._breaker_on_failure()

                if attempt_idx < attempts - 1:
                    await asyncio.sleep(self._calc_backoff(attempt_idx, backoff_base, backoff_cap, jitter))
                    continue
                else:
                    self._record_failure(e)
                    return self._decorate_fallback(
                        self.get_fallback_result(str(e)),
                        status=ComponentStatus.ERROR,
                        cache_hit=False,
                        attempts=attempt_idx + 1,
                        start_time=start_time
                    )


        self._record_failure(last_error or Exception("Unknown error"))
        return self._decorate_fallback(
            self.get_fallback_result("Unknown error"),
            status=ComponentStatus.ERROR,
            cache_hit=False,
            attempts=attempts,
            start_time=start_time
        )


    @abstractmethod
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        ...

    @abstractmethod
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        ...


    def _validate_inputs(self, inputs: Dict[str, Any]):
        return

    def _validate_output(self, output: Dict[str, Any]):
        return


    def _get_cache_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        return None

    def _auto_cache_key(self, inputs: Dict[str, Any]) -> Optional[str]:
        try:
            return f"{self.name}:{self._freeze(inputs)}"
        except Exception:
            return None

    def _cache_get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        entry = self._cache.get(key)
        if not entry:
            return None
        expiry, value = entry
        if now > expiry:

            try:
                del self._cache[key]
            except Exception:
                pass
            return None

        self._cache.move_to_end(key)
        return value

    def _cache_set(self, key: str, value: Dict[str, Any]):
        ttl = float(self.config.get('cache_ttl_sec', 0) or 0)
        if ttl <= 0:
            return
        expiry = time.time() + ttl
        self._cache[key] = (expiry, value)
        self._cache.move_to_end(key)

        max_entries = int(self.config.get('cache_max_entries', 128))
        while len(self._cache) > max_entries:
            self._cache.popitem(last=False)


    def _breaker_allows_call(self) -> bool:
        if not self.config.get('cb_enabled', True):
            return True

        if self._cb_state == 'closed':
            return True

        if self._cb_state == 'open':

            recovery = float(self.config.get('cb_recovery_time_sec', 15.0))
            if self._cb_opened_at is None:
                return False
            if (time.time() - self._cb_opened_at) >= recovery:
                self._cb_state = 'half_open'
                self._cb_half_open_calls = 0
                return True
            return False

        if self._cb_state == 'half_open':

            max_calls = int(self.config.get('cb_half_open_max_calls', 1))
            return self._cb_half_open_calls < max_calls

        return True

    def _breaker_on_call_begin(self):
        if not self.config.get('cb_enabled', True):
            return
        if self._cb_state == 'half_open':
            self._cb_half_open_calls += 1

    def _breaker_on_success(self):
        if not self.config.get('cb_enabled', True):
            return
        self._cb_state = 'closed'
        self._cb_failures = 0
        self._cb_opened_at = None
        self._cb_half_open_calls = 0

    def _breaker_on_failure(self):
        if not self.config.get('cb_enabled', True):
            return
        self._cb_failures += 1
        threshold = int(self.config.get('cb_failure_threshold', 5))
        if self._cb_failures >= threshold:
            self._cb_state = 'open'
            self._cb_opened_at = time.time()
            self.trace(f"Circuit opened after {self._cb_failures} failures", level="WARNING")


    def trace(self, message: str, level: str = "INFO", **kwargs):
        if self.logger:
            try:
                self.logger.trace(f"[{self.name}] {message}", level=level, **kwargs)
            except Exception:

                pass

    def _record_success(self, execution_time_ms: float):
        self._total_execution_time += execution_time_ms
        self._success_count += 1
        if self.metrics_tracker:
            try:
                self.metrics_tracker.record_success(f"{self.name}_analysis", execution_time_ms)
            except Exception:
                pass

    def _record_failure(self, exc: Exception):
        self._failure_count += 1
        self._last_error = str(exc)
        if self.metrics_tracker:
            try:
                self.metrics_tracker.record_failure(f"{self.name}_analysis", str(exc))
            except Exception:
                pass

    def _decorate_result(self, result: Dict[str, Any], start_time: float, cache_hit: bool, attempts: int) -> Dict[str, Any]:
        exec_ms = (time.time() - start_time) * 1000.0
        meta = {
            'name': self.name,
            'execution_time_ms': exec_ms,
            'cache_hit': cache_hit,
            'status': ComponentStatus.SUCCESS.value,
            'attempts': attempts,
            'circuit_state': self._cb_state
        }

        result_meta = result.get('_component_meta', {})
        result_meta.update(meta)
        result['_component_meta'] = result_meta
        return result

    def _decorate_fallback(
        self,
        fallback: Dict[str, Any],
        status: ComponentStatus,
        cache_hit: bool,
        attempts: int,
        start_time: float
    ) -> Dict[str, Any]:
        exec_ms = (time.time() - start_time) * 1000.0
        meta = {
            'name': self.name,
            'execution_time_ms': exec_ms,
            'cache_hit': cache_hit,
            'status': status.value,
            'attempts': attempts,
            'circuit_state': self._cb_state
        }
        fb_meta = fallback.get('_component_meta', {})
        fb_meta.update(meta)
        fallback['_component_meta'] = fb_meta

        if 'processing_success' not in fallback:
            fallback['processing_success'] = (status == ComponentStatus.SUCCESS)
        return fallback


    def get_state(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'config': self.config,
            'state': self._state,
            'metrics': {
                'execution_count': self._execution_count,
                'success_count': self._success_count,
                'failure_count': self._failure_count,
                'timeout_count': self._timeout_count,
                'average_execution_time': (
                    self._total_execution_time / max(self._success_count, 1)
                ),
                'circuit': {
                    'state': self._cb_state,
                    'failures': self._cb_failures,
                    'opened_at': self._cb_opened_at
                },
                'last_error': self._last_error
            }
        }

    def set_state(self, state: Dict[str, Any]):
        if 'state' in state:
            self._state = state['state']
        if 'metrics' in state:
            metrics = state['metrics']
            self._execution_count = metrics.get('execution_count', 0)
            self._success_count = metrics.get('success_count', 0)
            self._failure_count = metrics.get('failure_count', 0)
            self._timeout_count = metrics.get('timeout_count', 0)
            circ = metrics.get('circuit', {})
            self._cb_state = circ.get('state', 'closed')
            self._cb_failures = circ.get('failures', 0)
            self._cb_opened_at = circ.get('opened_at', None)
            self._last_error = metrics.get('last_error', None)

        self.trace(f"State restored for {self.name}", level="DEBUG")

    def clear_cache(self):
        self._cache.clear()
        self.trace(f"Cache cleared for {self.name}", level="DEBUG")

    def get_metrics(self) -> Dict[str, Any]:
        avg_time = self._total_execution_time / max(self._success_count, 1)
        return {
            'execution_count': self._execution_count,
            'success_count': self._success_count,
            'failure_count': self._failure_count,
            'timeout_count': self._timeout_count,
            'success_rate': (self._success_count / max(self._execution_count, 1)),
            'average_execution_time_ms': avg_time,
            'circuit_state': self._cb_state,
            'circuit_failures': self._cb_failures
        }

    def get_health_status(self) -> Dict[str, Any]:
        try:
            metrics = self.get_metrics()


            status = 'OK'
            is_healthy = True
            issues = []


            if not self.config.get('enabled', True):
                status = 'DISABLED'
                is_healthy = False
                issues.append("Component is disabled")


            if not self._initialized:
                status = 'ERROR'
                is_healthy = False
                issues.append("Component not initialized")


            if self._cb_state == 'open':
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"Circuit breaker open ({self._cb_failures} failures)")
            elif self._cb_state == 'half_open':
                status = 'WARNING'
                issues.append("Circuit breaker in half-open state (recovery probing)")


            success_rate = metrics['success_rate']
            if success_rate < 0.5 and self._execution_count > 5:
                status = 'DEGRADED'
                is_healthy = False
                issues.append(f"Low success rate: {success_rate:.1%}")
            elif success_rate < 0.8 and self._execution_count > 5:
                if status == 'OK':
                    status = 'WARNING'
                issues.append(f"Suboptimal success rate: {success_rate:.1%}")


            if self._timeout_count > 0 and self._execution_count > 0:
                timeout_rate = self._timeout_count / self._execution_count
                if timeout_rate > 0.2:
                    if status == 'OK':
                        status = 'WARNING'
                    issues.append(f"High timeout rate: {timeout_rate:.1%}")


            if self._last_error and status != 'DEGRADED':
                if status == 'OK':
                    status = 'WARNING'

            return {
                'status': status,
                'module': self.name,
                'version': '1.0',
                'is_healthy': is_healthy,
                'initialized': self._initialized,
                'enabled': self.config.get('enabled', True),
                'execution_count': self._execution_count,
                'success_count': self._success_count,
                'failure_count': self._failure_count,
                'timeout_count': self._timeout_count,
                'success_rate': success_rate,
                'circuit_state': self._cb_state,
                'circuit_failures': self._cb_failures,
                'last_error': self._last_error,
                'issues': issues if issues else None,
                'performance': {
                    'avg_time_ms': metrics['average_execution_time_ms'],
                    'total_executions': self._execution_count,
                    'cache_size': len(self._cache)
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': self.name,
                'version': '1.0',
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }


    def _calc_backoff(self, attempt_idx: int, base: float, cap: float, jitter: float) -> float:
        raw = min(cap, base * (2 ** attempt_idx))

        j = max(0.0, jitter)

        return max(0.0, raw + np.random.uniform(-j, j))

    def _freeze(self, obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, (str, int, float, bool)):
            return obj
        if isinstance(obj, dict):
            return tuple(sorted((k, self._freeze(v)) for k, v in obj.items()))
        if isinstance(obj, (list, tuple)):
            return tuple(self._freeze(x) for x in obj)

        if isinstance(obj, np.ndarray):
            return ('_nd_', int(obj.size), float(np.nan_to_num(obj.mean() if obj.size else 0.0)))

        return repr(obj)
