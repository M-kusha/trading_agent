# ─────────────────────────────────────────────────────────────
# File: modules/utils/circuit_breaker_utils.py
# Unified Circuit Breaker Utility - Single Source of Truth
# ─────────────────────────────────────────────────────────────

"""
Unified circuit breaker utility to replace manual implementations.

This module exports the production-grade CircuitBreaker from
modules/market/shared/circuit_breaker.py and provides helper functions
for common use cases.

Usage:
    from modules.utils.circuit_breaker_utils import CircuitBreaker, create_simple_breaker

    # Create a circuit breaker
    breaker = create_simple_breaker(name="MyModule", threshold=5, timeout=60.0)

    # Use in your module
    if breaker.allow_request():
        try:
            result = do_something()
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise
"""

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
    """
    Create a simple circuit breaker with basic settings.

    Args:
        name: Name for this breaker
        threshold: Consecutive failures before opening
        timeout: Seconds to stay open before trying half-open

    Returns:
        CircuitBreaker instance
    """
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
    """
    Create a standard circuit breaker with production settings.

    Args:
        name: Name for this breaker
        threshold: Consecutive failures before opening
        open_base_timeout: Base seconds open (with exponential backoff)
        open_max_timeout: Max seconds open
        window_seconds: Sliding window for failure rate
        failure_rate_threshold: Failure rate (0-1) to trip breaker

    Returns:
        CircuitBreaker instance with full features
    """
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
    """
    Migrate from old dict-based circuit breaker to new CircuitBreaker.

    Args:
        name: Name for the new breaker
        old_dict: Old dict with keys: state, failures, threshold, last_failure, cooldown_sec

    Returns:
        CircuitBreaker with state migrated from dict

    Example:
        # Old code:
        self.circuit_breaker = {
            "state": "CLOSED",
            "failures": 0,
            "threshold": 5,
            "last_failure": 0.0,
            "cooldown_sec": 300
        }

        # Migration:
        from modules.utils.circuit_breaker_utils import migrate_dict_breaker
        self.circuit_breaker = migrate_dict_breaker("DynamicRiskController", self.circuit_breaker)
    """
    threshold = int(old_dict.get('threshold', 5))
    cooldown = float(old_dict.get('cooldown_sec', 60.0))

    breaker = CircuitBreaker(
        name=name,
        threshold=threshold,
        timeout=cooldown,
        recovery_timeout=cooldown
    )

    # Restore state
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


# Helper function for replacing manual string-based breakers
def replace_manual_breaker_check(
    breaker: CircuitBreaker,
    cooldown_sec: float = 60.0
) -> bool:
    """
    Replace manual breaker state check with proper CircuitBreaker logic.

    Old pattern:
        if self._breaker_state == "OPEN":
            if (time.time() - self._last_failure_ts) >= cooldown_sec:
                self._breaker_state = "HALF_OPEN"
            return disabled_response()

    New pattern:
        if not replace_manual_breaker_check(self.circuit_breaker, cooldown_sec):
            return disabled_response()

    Args:
        breaker: CircuitBreaker instance
        cooldown_sec: Cooldown time (now handled by breaker internally)

    Returns:
        True if request should proceed, False if circuit is open
    """
    return breaker.allow_request()
