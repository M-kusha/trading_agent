# modules/memory/components/base.py

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

import numpy as np


class MemoryComponent(ABC):
    """Base class for all memory components."""

    # Names of shared resources expected to be present in `shared_resources`
    _REQUIRED_RESOURCES: tuple[str, ...] = (
        "store",
        "extractor",
        "scaler",
        "cache",
        "utils",
        "logger",
    )
    # Optional resources: component can degrade gracefully if absent
    _OPTIONAL_RESOURCES: tuple[str, ...] = ("pattern_detector", "encoder")

    # Default TTL (seconds) for cache entries unless caller overrides
    _DEFAULT_CACHE_TTL: int = 60

    def __init__(self, config: Any, shared_resources: Dict[str, Any]) -> None:
        """
        Initialize a memory component.

        Args:
            config: UnifiedMemoryConfig (or compatible) instance.
            shared_resources: A dict with common resources provisioned by UnifiedMemory:
                - REQUIRED: 'store', 'extractor', 'scaler', 'cache', 'utils', 'logger'
                - OPTIONAL: 'pattern_detector', 'encoder'
        Raises:
            ValueError: if any required shared resource is missing.
        """
        self.config = config
        self.shared = shared_resources

        # Validate and extract resources
        missing = [k for k in self._REQUIRED_RESOURCES if k not in shared_resources]
        if missing:
            raise ValueError(
                f"{self.__class__.__name__}: missing required shared resources: {missing}"
            )

        # Extract commonly used resources (typed as Any to keep this base dependency-light)
        self.store = shared_resources["store"]
        self.extractor = shared_resources["extractor"]
        self.pattern_detector = shared_resources.get("pattern_detector")
        self.scaler = shared_resources["scaler"]
        self.encoder = shared_resources.get("encoder")
        self.cache = shared_resources["cache"]
        self.utils = shared_resources["utils"]
        self.logger = shared_resources["logger"]

        # Component state
        self._initialized: bool = False
        self._last_process_s: float = 0.0
        self._process_count: int = 0
        self._error_count: int = 0

        # Allow subclasses to prepare their internals
        self._initialize_component()
        self._initialized = True

        # Informational log (quiet if debug disabled)
        self._log_debug("initialized", details={"requires": self._REQUIRED_RESOURCES})

    # -------------------------------------------------------------------------
    # Abstract hooks
    # -------------------------------------------------------------------------

    @abstractmethod
    def _initialize_component(self) -> None:
        """Subclasses set up any component-specific resources here."""
        raise NotImplementedError

    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform the component's compute step.

        Args:
            context: A dictionary with inputs (market data, trades, features, etc.)
        Returns:
            A dictionary of component-specific outputs to be merged upstream.
        """
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    def get_relevant_memories(
        self,
        query: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories from the shared store.

        Args:
            query: Query vector (numpy array).
            k: Number of memories to retrieve.
            filter_fn: Optional predicate applied to each candidate memory.
        Returns:
            List of memory entries (dicts) from the store.
        Raises:
            TypeError: if `query` is not a numpy array.
            ValueError: if `k` is non-positive.
        """
        if not isinstance(query, np.ndarray):
            raise TypeError(
                f"{self.__class__.__name__}.get_relevant_memories: 'query' must be np.ndarray, "
                f"got {type(query).__name__}"
            )
        if k <= 0:
            raise ValueError(
                f"{self.__class__.__name__}.get_relevant_memories: 'k' must be > 0, got {k}"
            )

        self._log_debug(
            "get_relevant_memories",
            details={"k": k, "query_shape": tuple(query.shape)},
        )
        return self.store.query(query, k, filter_fn)

    def cache_get(self, key: str) -> Optional[Any]:
        """
        Namespaced cache read for this component.

        Args:
            key: Logical cache key.
        Returns:
            Cached value or None.
        """
        namespaced = f"{self.__class__.__name__}:{key}"
        value = self.cache.get(namespaced)
        self._log_debug("cache_get", details={"key": namespaced, "hit": value is not None})
        return value

    def cache_put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Namespaced cache write for this component.

        Args:
            key: Logical cache key.
            value: Value to cache.
            ttl: Time-to-live in seconds; defaults to component default.
        """
        namespaced = f"{self.__class__.__name__}:{key}"
        ttl_s = int(ttl if ttl is not None else self._DEFAULT_CACHE_TTL)
        self.cache.put(namespaced, value, ttl_s)
        self._log_debug("cache_put", details={"key": namespaced, "ttl": ttl_s})

    def mark_process_start(self) -> float:
        """Mark the start of a processing window (returns a monotonic start timestamp)."""
        return perf_counter()

    def mark_process_end(self, start_ts: float) -> None:
        """
        Mark the end of processing, updating timing and counters.

        Args:
            start_ts: Start timestamp returned by `mark_process_start`.
        """
        duration = max(0.0, perf_counter() - start_ts)
        self._last_process_s = duration
        self._process_count += 1
        self._log_debug(
            "process_complete",
            details={"duration_ms": round(duration * 1000, 3), "count": self._process_count},
        )

    def log_error(self, message: str, error: Exception) -> None:
        """
        Log an error and increment the component error counter.

        Args:
            message: Contextual message.
            error: Exception instance that occurred.
        """
        self._error_count += 1
        try:
            self.logger.error(f"[{self.__class__.__name__}] {message}: {error}")
        except Exception:
            # Guard against unexpected logger signatures
            pass

    def get_stats(self) -> Dict[str, Any]:
        """
        Snapshot of component health and activity.

        Returns:
            Dict with initialized flag, last process duration (s), process & error counts.
        """
        return {
            "initialized": self._initialized,
            "last_process_s": self._last_process_s,
            "process_count": self._process_count,
            "error_count": self._error_count,
        }

    # -------------------------------------------------------------------------
    # Internal logging helper
    # -------------------------------------------------------------------------

    def _log_debug(self, message: str, *, details: Optional[Dict[str, Any]] = None) -> None:
        """Internal: guarded debug log with optional structured details."""
        if not getattr(self.config, "debug", False):
            return
        try:
            if details:
                self.logger.debug(f"[{self.__class__.__name__}] {message} | {details}")
            else:
                self.logger.debug(f"[{self.__class__.__name__}] {message}")
        except Exception:
            # Be resilient to logger API differences
            pass
