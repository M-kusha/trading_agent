

from typing import Any, Dict

from modules.market.shared.circuit_breaker import CircuitBreaker, CircuitState

__all__ = [
    'CircuitBreaker',
    'CircuitState',
    'create_simple_breaker',
    'create_standard_breaker',
    'migrate_dict_breaker',
]


def create_simple_breaker(
    name: str,
    threshold: int = 5,
    timeout: float = 60.0
) -> CircuitBreaker:
    return CircuitBreaker(
        name=name,
        threshold=threshold,
        timeout=timeout,
        recovery_timeout=timeout
    )


def create_standard_breaker(
    name: str,
    threshold: int = 5,
    open_base_timeout: float = 15.0,
    open_max_timeout: float = 120.0,
    window_seconds: float = 30.0,
    failure_rate_threshold: float = 0.5
) -> CircuitBreaker:
    return CircuitBreaker(
        name=name,
        threshold=threshold,
        open_base_timeout=open_base_timeout,
        open_max_timeout=open_max_timeout,
        window_seconds=window_seconds,
        failure_rate_threshold=failure_rate_threshold,
        half_open_max_calls=1,
        half_open_successes_to_close=1,
        jitter_fraction=0.1
    )


def migrate_dict_breaker(
    name: str,
    old_dict: Dict[str, Any]
) -> CircuitBreaker:
    threshold = int(old_dict.get('threshold', 5))
    cooldown = float(old_dict.get('cooldown_sec', 60.0))

    breaker = CircuitBreaker(
        name=name,
        threshold=threshold,
        timeout=cooldown,
        recovery_timeout=cooldown
    )


    old_state = old_dict.get('state', 'CLOSED')
    failures = int(old_dict.get('failures', 0))
    last_failure = float(old_dict.get('last_failure', 0.0))

    state_dict = {
        'state': old_state.lower() if isinstance(old_state, str) else 'closed',
        'failure_count': failures,
        'failure_streak': failures,
        'last_failure_time': last_failure,
    }

    breaker.set_state(state_dict)

    return breaker


def replace_manual_breaker_check(
    breaker: CircuitBreaker,
    cooldown_sec: float = 60.0
) -> bool:
    return breaker.allow_request()
